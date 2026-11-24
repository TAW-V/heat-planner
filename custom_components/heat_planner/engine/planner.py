"""Full planning run: raw data -> inputs -> hot water plan -> heating (+ battery) plan -> result dictionary.

Read-only: the result is only published as sensor attributes. Nothing here talks to devices.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import time
from zoneinfo import ZoneInfo

import numpy as np

from . import accounting as A
from . import inputs as I
from . import params as P
from . import profile as HP
from .dhw import plan_dhw, schedule_dhw
from .model import DT, Model, dhw_charge, dhw_cop
from .optimizer import battery_options, blocks, elec_cost, plan

_LOGGER = logging.getLogger(__name__)
STEP = I.STEP
HIST_DAYS_STATE = 2       # days of 5-minute history for the state estimate
DHW_RISE_EVIDENCE = 1.0   # K tank rise since the start of a planned charge that shows it runs (besides the setpoint)
ROOM_RANGE = (0.0, 40.0)         # degC, plausible room temperature samples
OUTDOOR_RANGE = (-50.0, 60.0)    # degC, plausible outdoor temperature samples
HEAT_SPIKE_FACTOR = 2.0          # heat output samples above this x the maximum heat output are invalid (e.g. 6553 kW)
# plausibility of the estimated state [Tr, Ts, d]; outside it the estimate is rebuilt (see state_estimate)
STATE_MAX_D = 3.0                # kW, |unmeasured gains|
STATE_TS_TR = (-3.0, 8.0)        # K, building mass minus room air
STATE_MAX_ROOM_DEV = 0.5         # K, |estimated - measured room temperature|
STATE_RESET_HOURS = 12           # h of history for the second attempt
# options echoed in the plan sensor (so dashboards can show the active settings)
SETTINGS_SHOWN = (P.MIN_TEMP, P.MAX_TEMP, P.HARD_MIN_TEMP, P.SAFETY_MARGIN, P.MARGIN_PENALTY, P.COMFORT_PENALTY,
                  P.MIN_BLOCK_HOURS, P.START_COST, P.TAIL_HOURS, P.AVOID_PEAKS, P.PLAN_BATTERY, P.DHW_MODE, P.HEAT_LOSS, P.MASS_CAPACITY)


class PlanError(Exception):
    """input missing or unusable"""


def _iso(x: float, tz: ZoneInfo, spec: str = "auto") -> str:
    return dt.datetime.fromtimestamp(x, tz).isoformat(timespec=spec)


def _r(x, n=2):
    return None if x is None else round(float(x), n)


def _live_power(p: dict, st: dict, eid_key: str, attr_key: str):
    """(kW, source) from an entity state or attribute (W/kW by the entity's unit; without a unit, values above 100 are W);
    None when not configured, missing or not numeric"""
    eid = p.get(eid_key)
    if not eid or eid not in st:
        return None
    attr = (p.get(attr_key) or "").strip()
    v = I.num((st[eid].get("a") or {}).get(attr) if attr else st[eid].get("s"))
    if v is None:
        return None
    v = abs(v)
    unit = I.state_unit(st, eid)
    v = v * I.unit_factor(unit, "power") if unit else (v / 1000 if v > 100 else v)
    return v, (f"{eid}.{attr}" if attr else eid)


def live_dhw_low(p: dict, st: dict) -> dict:
    """low tank setpoint from the optional entity (e.g. the base setpoint helper an automation writes to the heat pump);
    unavailable or non-numeric values keep the configured number"""
    eid = p.get(P.DHW_LOW_ENTITY)
    v = I.temp_c(I.state_num(st, eid), I.state_unit(st, eid)) if eid else None
    if v is None:
        return {}
    _, lo, hi = P.NUMBERS[P.DHW_LOW]
    p[P.DHW_LOW] = min(max(v, lo), hi)
    return {P.DHW_LOW: eid}


def live_battery(p: dict, st: dict) -> dict:
    """capacity, power limits and min/max charge from the optional source entities (state or attribute), in place of the numbers.
    W/Wh are converted by the unit of the entity; without a unit, values above 100 are taken as W/Wh.
    Unavailable or non-numeric values keep the configured number. Returns {key: source} for the live values.
    The optional live power limits for the current quarter are stored in p['_live_charge_kw'] / p['_live_discharge_kw']
    (min(planning limit, live value); None = not configured or unavailable)."""
    used = {}
    eid = p.get(P.BAT_CAPACITY_ENTITY)
    if eid and eid in st:
        v = I.num(st[eid].get("s"))
        if v is not None:
            v = abs(v)
            unit = I.state_unit(st, eid)
            v = v * I.unit_factor(unit, "energy") if unit else (v / 1000 if v > 100 else v)
            if v > 0:
                _, lo, hi = P.NUMBERS[P.BAT_CAPACITY]
                p[P.BAT_CAPACITY] = min(max(v, lo), hi)
                used[P.BAT_CAPACITY] = eid
    for key, eid_key, attr_key in ((P.BAT_CHARGE_KW, P.BAT_CHARGE_POWER_ENTITY, P.BAT_CHARGE_POWER_ATTR),
                                   (P.BAT_DISCHARGE_KW, P.BAT_DISCHARGE_POWER_ENTITY, P.BAT_DISCHARGE_POWER_ATTR)):
        r = _live_power(p, st, eid_key, attr_key)
        if r is None:
            continue
        _, lo, hi = P.NUMBERS[key]
        p[key] = min(max(r[0], lo), hi)
        used[key] = r[1]
    for key, plan_key, eid_key, attr_key in (
            ("_live_charge_kw", P.BAT_CHARGE_KW, P.BAT_LIVE_CHARGE_POWER_ENTITY, P.BAT_LIVE_CHARGE_POWER_ATTR),
            ("_live_discharge_kw", P.BAT_DISCHARGE_KW, P.BAT_LIVE_DISCHARGE_POWER_ENTITY, P.BAT_LIVE_DISCHARGE_POWER_ATTR)):
        r = _live_power(p, st, eid_key, attr_key)
        p[key] = None if r is None else min(r[0], p[plan_key])
        if r is not None:
            used[key[1:]] = r[1]
    # minimum / maximum charge in % of the capacity (entities in kWh/Wh are converted with the capacity used above)
    for key, ids, pick in ((P.BAT_MIN_SOC, p.get(P.BAT_MIN_SOC_ENTITIES) or [], max),
                           (P.BAT_MAX_SOC, [p.get(P.BAT_MAX_SOC_ENTITY)], min)):
        vals = {}
        for eid in ids:
            v = I.num((st.get(eid) or {}).get("s")) if eid else None
            if v is None:
                continue
            unit = (I.state_unit(st, eid) or "%").strip().lower()
            if unit != "%":
                v = v * I.unit_factor(unit, "energy") / p[P.BAT_CAPACITY] * 100
            vals[eid] = min(max(v, 0.0), 100.0)
        if vals:
            p[key] = pick(vals.values())
            used[key] = list(vals) if len(vals) > 1 else next(iter(vals))
    if p[P.BAT_MAX_SOC] < p[P.BAT_MIN_SOC]:
        p[P.BAT_MAX_SOC] = p[P.BAT_MIN_SOC]
    return used


def _plausible_rows(rows, lo: float, hi: float, conv=lambda v: v) -> tuple[list, int]:
    """(5-minute statistic rows whose mean/min/max all lie within [lo, hi] after conversion, number of rows dropped)"""
    keep, dropped = [], 0
    for r in rows or []:
        vals = [I.num(r.get(k)) for k in ("mean", "min", "max") if r.get(k) is not None]
        if vals and any(v is None or not lo <= conv(v) <= hi for v in vals):
            dropped += 1
            continue
        keep.append(r)
    return keep, dropped


def _temp_conv(unit: str | None):
    return (lambda v: (v - 32) / 1.8) if (unit or "").upper().endswith("F") else (lambda v: v)


def history(raw: dict, p: dict, t, tz: ZoneInfo, prof: dict | None = None, info: dict | None = None):
    """15-min history arrays (room, outdoor, space-heating heat, PV, household electricity) over grid t.

    Implausible samples are dropped before averaging (e.g. ebusd reports 0xFFFF/10 = 6553 kW while a value is invalid):
    room temperature outside ROOM_RANGE and outdoor temperature outside OUTDOOR_RANGE are left out (the Kalman filter skips
    quarters without a room value; the outdoor temperature is interpolated). Heat output below 0 or above
    HEAT_SPIKE_FACTOR x the maximum heat output is left out; a quarter that loses all its samples that way is linearly
    interpolated between its neighbours (the heat pump was most likely running), while quarters without any data stay 0.
    info (optional dict) receives {"dropped": {series: number of 5-minute samples left out}}."""
    st = raw["states"]
    s5 = raw.get("stats_5min") or {}
    room_id, out_id = p.get(P.ROOM_TEMP_ENTITY), p.get(P.OUTDOOR_TEMP_ENTITY)
    dropped = {}
    cr = _temp_conv(I.state_unit(st, room_id))
    rows, dropped["room"] = _plausible_rows(s5.get(room_id), *ROOM_RANGE, cr)
    Tr = cr(I.series_15min(rows, t))
    co = _temp_conv(I.state_unit(st, out_id))
    rows, dropped["outdoor"] = _plausible_rows(s5.get(out_id), *OUTDOOR_RANGE, co)
    To = co(I.series_15min(rows, t))
    hid = p.get(P.HEAT_OUTPUT_ENTITY)
    dropped["heat"] = 0
    if hid:
        mask = s5.get(p.get(P.HEATING_ACTIVE_ENTITY)) if p.get(P.HEATING_ACTIVE_ENTITY) else None
        rows_all = s5.get(hid) or []
        unit = I.state_unit(st, hid)
        if unit:
            f = I.unit_factor(unit, "power")
        else:   # no unit: W when typical (median of the positive values) readings are above 50
            pos = [v for v in (I.num(r.get("mean")) for r in rows_all) if v is not None and 0.1 < v < 6553]
            f = 1 / 1000 if pos and float(np.median(pos)) > 50 else 1.0
        rows, dropped["heat"] = _plausible_rows(rows_all, 0.0, HEAT_SPIKE_FACTOR * p[P.HP_MAX_HEAT], lambda v: v * f)
        Qy = I.series_15min(rows, t, mask_rows=mask) * f
        if dropped["heat"]:
            had = ~np.isnan(I.series_15min(rows_all, t))
            gap = had & np.isnan(Qy)        # quarters that only had implausible samples
            Qy = np.nan_to_num(Qy)
            ok = ~gap
            if gap.any() and ok.any():
                Qy[gap] = np.interp(np.flatnonzero(gap), np.flatnonzero(ok), Qy[ok])
    else:
        Qy = np.zeros(len(t))
    if info is not None:
        info["dropped"] = {k: v for k, v in dropped.items() if v}
    pid = p.get(P.PV_POWER_ENTITY)
    PV = np.nan_to_num(I.series_15min(s5.get(pid), t)) * I.unit_factor(I.state_unit(st, pid), "power") if pid else np.zeros(len(t))
    Eh = house_energy(raw, p)
    Ep = HP.series(prof, t, tz) if prof is not None else np.full(len(t), float(np.median(list(Eh.values()))) if Eh else 0.0)
    E = np.array([Eh.get(x // 3600 * 3600, Ep[i]) for i, x in enumerate(t)])
    ok = ~np.isnan(To)
    To = np.interp(np.arange(len(t)), np.flatnonzero(ok), To[ok]) if ok.any() else np.full(len(t), 8.0)
    return Tr, To, np.nan_to_num(Qy), PV, E


def _state_problem(x0, room: float | None) -> str | None:
    """why an estimated state [Tr, Ts, d] is implausible, None when it is fine"""
    if not np.all(np.isfinite(x0)):
        return "not finite"
    if abs(x0[2]) > STATE_MAX_D:
        return f"gains {x0[2]:.1f} kW"
    if not STATE_TS_TR[0] <= x0[1] - x0[0] <= STATE_TS_TR[1]:
        return f"mass-room {x0[1] - x0[0]:.1f} K"
    if room is not None and abs(x0[0] - room) > STATE_MAX_ROOM_DEV:
        return f"room {x0[0]:.2f} vs measured {room:.2f}"
    return None


def state_estimate(model: Model, U: np.ndarray, Tr_h, room: float | None = None):
    """(x0 at the start of the next slot, reset reason or None).
    Kalman filter over the whole history; if the result is implausible (see _state_problem) it is filtered again over the
    last STATE_RESET_HOURS only, and if that is implausible too the state is set to Tr = measured room, Ts = Tr + 0.5, d = 0."""
    Tr_h = np.asarray(Tr_h, float)
    if room is None:
        good = Tr_h[~np.isnan(Tr_h)]
        room = float(good[-1]) if good.size else None
    X = model.filter(U, Tr_h)
    x0 = model.Ad @ X[-1] + model.Bd @ U[-1]
    why = _state_problem(x0, room)
    if why is None:
        return x0, None
    n = int(STATE_RESET_HOURS / DT)
    if len(Tr_h) > n and not np.all(np.isnan(Tr_h[-n:])):
        X = model.filter(U[-n:], Tr_h[-n:])
        x1 = model.Ad @ X[-1] + model.Bd @ U[-1]
        if _state_problem(x1, room) is None:
            return x1, f"{why}; filtered over the last {STATE_RESET_HOURS} h"
    r = room if room is not None else float(x0[0]) if np.isfinite(x0[0]) else 20.0
    return np.array([r, r + 0.5, 0.0]), f"{why}; reset to the measured room temperature"


def energy_ids(p: dict) -> dict:
    """configured energy counters of the household balance, keyed by role"""
    return {role: p.get(key) for role, key in P.ENERGY_ROLES.items() if p.get(key)}


def energy_scales(st: dict, ids: dict) -> dict:
    return {k: I.unit_factor(I.state_unit(st, v), "energy") for k, v in ids.items()}


def house_energy(raw: dict, p: dict) -> dict:
    """household kWh per hour from the hourly statistics (battery counters missing = 0)"""
    ids = energy_ids(p)
    return I.hourly_house_energy(raw.get("stats_hour") or {}, ids, energy_scales(raw["states"], ids))


def build_house_profile(raw: dict, p: dict, archive: dict | None = None) -> tuple[dict, dict]:
    """household profile per weekday and quarter + updated quarter archive.
    raw: now, tz, states, stats_hour (hourly counters, as many weeks as available) and optionally stats_5min_energy
    (5-minute counters of the last days, for the quarter shape)."""
    tz = ZoneInfo(raw.get("tz") or "UTC")
    now = raw["now"]
    ids = energy_ids(p)
    sc = energy_scales(raw["states"], ids)
    hourly = HP.house_changes(raw.get("stats_hour") or {}, ids, sc)
    five = HP.house_changes(raw.get("stats_5min_energy") or {}, ids, sc, period_s=300, drop_first=True)
    arch = HP.merge_archive(archive, HP.quarter_days(five, tz, HP._midnight(now, tz)), now, tz)
    return HP.build(hourly, arch, now, tz), arch


def house_profile(raw: dict, p: dict, tz: ZoneInfo) -> dict:
    """profile prepared by the coordinator (once a day, many weeks), else built from the statistics in this run"""
    prof = raw.get("house_profile")
    if HP.valid(prof):
        return prof
    prof, _ = build_house_profile(raw, p, raw.get("quarter_archive"))
    prof["source"] = "run"
    return prof


def price_settings(raw: dict, p: dict) -> dict:
    """net metering on/off, live net metering fee and fixed costs (taxes) where entities provide them"""
    st = raw["states"]
    nm = I.state_on(st, p.get(P.NET_METERING_ENTITY))
    net_metering = p[P.NET_METERING] if nm is None else nm
    vat = p[P.VAT_PERCENT] / 100
    K = p[P.EXPORT_FIXED_COSTS]
    fee = p[P.NET_METERING_FEE]
    bid = p.get(P.PRICE_BREAKDOWN_ENTITY)
    allin = None
    if bid:
        f = I.unit_factor(I.state_unit(st, bid), "price")
        allin = I.state_num(st, bid)
        energy = I.num(((st.get(bid) or {}).get("a") or {}).get("energy_price"))
        if allin is not None and energy is not None:
            K = (allin - energy * (1 + vat)) * f
        allin = allin * f if allin is not None else None
    eid = p.get(P.EXPORT_PRICE_ENTITY)
    if eid:
        if allin is None:
            allin = I.state_num(st, p.get(P.PRICE_ENTITY))
            allin = allin * I.unit_factor(I.state_unit(st, p.get(P.PRICE_ENTITY)), "price") if allin is not None else None
        ex = I.state_num(st, eid)
        if allin is not None and ex is not None:
            v = allin - ex * I.unit_factor(I.state_unit(st, eid), "price")
            if 0 <= v <= 0.15:
                fee = v
    return dict(net_metering=bool(net_metering), K=K, vat=vat, fee=fee)


def build_inputs(raw: dict, p: dict, tz: ZoneInfo, t0: int, prof: dict | None = None):
    """15-min input arrays from t0 to (last known price + tail)"""
    st = raw["states"]
    now = raw["now"]
    Pd = I.price_dict(raw.get("prices"))
    Pd = {k: v for k, v in Pd.items() if k >= t0}
    if not Pd:
        raise PlanError("no known prices")
    tk_end = max(Pd) + STEP
    tail_h = p[P.TAIL_HOURS]
    t = np.arange(t0, tk_end + tail_h * 3600, STEP, dtype=float)
    N = len(t)
    pid = p.get(P.PRICE_ENTITY)
    pf = I.unit_factor(I.state_unit(st, pid), "price")
    Phist = {k: v * pf for k, v in I.stat_series((raw.get("stats_hour") or {}).get(pid), "mean").items()}
    if not Phist:   # no price statistics: use the known prices themselves for the expected-price profile
        Phist = I.price_dict(raw.get("prices"))
    Pmed = I.median_by_hour(Phist, now, tz)
    hh = np.array([dt.datetime.fromtimestamp(x, tz).hour for x in t])
    price = np.array([Pd.get(x, np.nan) for x in t])
    known = ~np.isnan(price)
    price[~known] = np.array(Pmed)[hh[~known]]
    ps = price_settings(raw, p)
    fc_info = None
    fid = p.get(P.PRICE_FORECAST_ENTITY)
    if fid and (~known).any():
        fst = st.get(fid) or {}
        spot = I.forecast_price_dict(fst.get("a"), (fst.get("a") or {}).get("unit_of_measurement"))
        if spot:
            adder, n_cal = I.forecast_adder(I.price_dict(raw.get("prices")), spot, ps["vat"], ps["K"])
            use = np.array([(not known[i]) and (x in spot) for i, x in enumerate(t)])
            if use.any():
                price[use] = np.array([spot[x] for x in t[use]]) * (1 + ps["vat"]) + adder
            fc_info = dict(entity=fid, slots=int(use.sum()), adder=round(adder, 4), calibration_slots=n_cal,
                           until=_iso(float(t[use].max()) + STEP, tz) if use.any() else None)
    E = HP.series(prof if prof is not None else house_profile(raw, p, tz), t, tz)
    wid = p.get(P.WEATHER_ENTITY)
    if raw.get("weather"):
        To = I.weather_series(raw["weather"], t, tz, p["_bias"], ((st.get(wid) or {}).get("a") or {}).get("temperature_unit"))
    else:
        cur = I.temp_c(I.state_num(st, p.get(P.OUTDOOR_TEMP_ENTITY)), I.state_unit(st, p.get(P.OUTDOOR_TEMP_ENTITY)))
        if cur is None:
            raise PlanError("no weather forecast and no outdoor temperature")
        To = np.full(N, cur - 1.0)
    PV = I.pv_forecast_series(st, p.get(P.PV_FORECAST_ENTITIES) or [], t)
    pk = np.zeros(N, bool)
    if p[P.AVOID_PEAKS]:
        if p.get(P.PEAK_ENTITY):
            for a, b in I.peak_periods((st.get(p[P.PEAK_ENTITY]) or {}).get("a")):
                pk |= (t >= a) & (t < b)
            tf = I.top_share_peaks(t, price, tz, p[P.PEAK_SHARE])
            pk[~known] = tf[~known]
        else:
            pk = I.top_share_peaks(t, price, tz, p[P.PEAK_SHARE])
    feed = I.feed_prices(price, ps["net_metering"], ps["K"], ps["vat"], ps["fee"], p[P.EXPORT_FEE])
    return dict(t=t, hours=(t - t0) / 3600, To=To, PV=PV, E=E, price=price, feed=feed, peak=pk, known=known,
                dhw=np.zeros(N, bool), dhw_kw=np.zeros(N), price_forecast=fc_info), ps, tk_end


def legionella_busy(raw: dict, p: dict, t) -> np.ndarray:
    st = raw["states"]
    lid = p.get(P.LEGIONELLA_ENTITY)
    busy = np.zeros(len(t), bool)
    if lid and I.state_on(st, lid):
        since = I.num((st.get(lid) or {}).get("lc"), raw["now"])
        busy |= (np.asarray(t) + STEP > since) & (np.asarray(t) < since + p[P.LEGIONELLA_HOURS] * 3600)
    return busy


def tank_status(raw: dict, p: dict, tz: ZoneInfo):
    """(charges done today, end of the last charge, current tank temperature) from the tank sensor"""
    tid = p.get(P.DHW_TANK_ENTITY)
    if not tid:
        return 0, None, None
    st = raw["states"]
    unit = I.state_unit(st, tid)
    cur = I.temp_c(I.state_num(st, tid), unit)
    thr = p[P.DHW_TARGET] - p[P.DHW_HYST]
    now = raw["now"]
    midnight = dt.datetime.fromtimestamp(now, tz).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    done, last = 0, None
    for r in (raw.get("stats_5min") or {}).get(tid, []):
        v = I.temp_c(I.num(r.get("max", r.get("mean"))), unit)
        s = r["start"] / 1000
        if v is not None and v >= thr and s >= now - 86400:
            last = s + 300
            if s >= midnight:
                done = 1
    if cur is not None and cur >= thr:
        done = 1
    return done, last, cur


def _parse_time(v, tz: ZoneInfo) -> float | None:
    """input_datetime state ('YYYY-MM-DD HH:MM:SS', local), ISO timestamp or epoch seconds -> epoch seconds"""
    if v in (None, "", "unknown", "unavailable"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    try:
        d = dt.datetime.fromisoformat(str(v).strip().replace(" ", "T") if "T" not in str(v) else str(v).strip())
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=tz)
    return d.timestamp()


def legionella_status(raw: dict, p: dict, tz: ZoneInfo) -> dict | None:
    """last time the tank reached legionella_temp (restart-proof: entity, long-term hourly maxima, 5-minute maxima, current
    value; the latest wins) and the resulting earliest/due times. None when legionella_mode is not 'plan'.
    No information at all = due now (a legionella charge is planned within 24 h)."""
    if p[P.LEGIONELLA_MODE] != "plan":
        return None
    st = raw["states"]
    now = raw["now"]
    thr = p[P.LEGIONELLA_TEMP] - 0.05
    tid = p.get(P.DHW_TANK_ENTITY)
    unit = I.state_unit(st, tid) if tid else None
    cands, src = [], []
    lid = p.get(P.LEGIONELLA_LAST_ENTITY)
    if lid:
        v = _parse_time((st.get(lid) or {}).get("s"), tz)
        if v is not None and v <= now + 60:
            cands.append(v)
            src.append("entity")
    for key, length in (("stats_tank_hour", 3600), ("stats_5min", 300)):
        rows = (raw.get(key) or {}).get(tid, []) if tid else []
        hot = [r["start"] / 1000 + length for r in rows
               if (I.temp_c(I.num(r.get("max", r.get("mean"))), unit) or -99) >= thr]
        if hot:
            cands.append(min(max(hot), now))
            src.append("statistics")
    cur = I.temp_c(I.state_num(st, tid), unit) if tid else None
    if cur is not None and cur >= thr:
        cands.append(now)
        src.append("tank")
    last = max(cands) if cands else None
    day = 86400.0
    due = (last + p[P.LEGIONELLA_INTERVAL] * day) if last is not None else now
    earliest = (last + p[P.LEGIONELLA_EARLIEST] * day) if last is not None else now
    return dict(last=last, due=due, earliest=earliest, overdue=due <= now, source=src[cands.index(last)] if cands else None)


def dhw_pause(raw: dict, p: dict, tz: ZoneInfo) -> float | None:
    """end of a pause of the hot-water execution (optional entity, e.g. set by an automation after a manual setpoint change);
    None when not configured, unavailable or already over"""
    eid = p.get(P.DHW_PAUSE_ENTITY)
    v = _parse_time((raw["states"].get(eid) or {}).get("s"), tz) if eid else None
    return v if v is not None and v > raw["now"] else None


def dhw_setpoint(p: dict, st: dict) -> float | None:
    """live tank setpoint of the heat pump (degC); None when not configured or unavailable"""
    eid = p.get(P.DHW_SETPOINT_ENTITY)
    return I.temp_c(I.state_num(st, eid), I.state_unit(st, eid)) if eid else None


def _tank_rise(raw: dict, p: dict, since: float, tank: float | None) -> float:
    """K the tank temperature rose since `since` (current value - lowest 5-minute mean from 10 minutes before `since`)"""
    tid = p.get(P.DHW_TANK_ENTITY)
    if tank is None or not tid:
        return 0.0
    unit = I.state_unit(raw["states"], tid)
    vals = [I.temp_c(I.num(r.get("mean", r.get("max"))), unit) for r in (raw.get("stats_5min") or {}).get(tid, [])
            if r["start"] / 1000 >= since - 600]
    vals = [v for v in vals if v is not None]
    return tank - min(vals) if vals else 0.0


def dhw_running(raw: dict, p: dict, previous: dict | None, tank: float | None, sp: float | None, pause: float | None,
                legionella_busy_now: bool) -> tuple[tuple | None, dict]:
    """the planned charge that runs now: (start, end, kind, setpoint cap) for plan_dhw or None, and the check result.

    A charge of the previous plan whose start has passed only counts as running (and may be extended) when the heat pump
    shows it: its tank setpoint (dhw_setpoint_entity) is above the low setpoint, or the tank warmed by
    DHW_RISE_EVIDENCE K since the start to above the low setpoint. Without that evidence it counts as running only during
    dhw_start_grace_minutes after its start (the executor writes the setpoint at the quarter); afterwards it was not
    executed and the charge is planned again. Once confirmed it stays running until its end (a glitch in the setpoint
    reading cannot end a real charge). While the hot-water execution is paused the executor does not hold the charge:
    it then only runs while the setpoint is raised and the tank is below it. Without a setpoint entity (or while it is
    unavailable) every started charge counts as running, as before.
    After a restart (no previous plan) a raised setpoint with the tank below it is taken as a running charge."""
    now = raw["now"]
    low = p[P.DHW_LOW]
    raised = sp is not None and sp > low + 0.5
    cand = None
    for b in ((previous or {}).get("_dhw_blocks") or []):
        if b[0] <= now < b[1]:
            cand = b
    if cand is None:
        if previous is None and raised and tank is not None and tank < sp - 0.5 and not legionella_busy_now:
            lc = I.num((raw["states"].get(p.get(P.DHW_SETPOINT_ENTITY)) or {}).get("lc"), now)
            if now - lc < p[P.DHW_MAX_CHARGE_HOURS] * 3600:
                kind = ("legionella" if sp >= p[P.LEGIONELLA_SETPOINT] - 0.5 else
                        "hot" if p[P.DHW_HOT_PRICE_RATIO] > 0 and sp >= p[P.DHW_HOT_TARGET] - 0.5 else "normal")
                a = min(lc, now)
                return (a, I.q0(now) + STEP, kind, sp), dict(state="restored", setpoint=sp, start=a, confirmed=True, since=a)
        return None, dict(state=None, setpoint=sp)
    a, b = float(cand[0]), float(cand[1])
    kind = cand[2] if len(cand) > 2 and cand[2] else "normal"
    confirmed = bool(cand[3]) if len(cand) > 3 else False
    since = float(cand[4]) if len(cand) > 4 and cand[4] is not None else a
    if sp is None:                      # no evidence available: as before
        return (a, b, kind), dict(state="unchecked", setpoint=None, start=a, confirmed=confirmed, since=since)
    cap = sp if raised else None
    rise = _tank_rise(raw, p, a, tank)
    evidence = raised or (rise >= DHW_RISE_EVIDENCE and tank is not None and tank > low + 0.5)
    if pause is not None:
        if raised and (tank is None or tank < sp):
            return (a, b, kind, cap), dict(state="paused_running", setpoint=sp, start=a, confirmed=True, since=since)
        return None, dict(state="paused", setpoint=sp, start=a, confirmed=False, since=since)
    if confirmed or evidence:
        return (a, b, kind, cap), dict(state="confirmed", setpoint=sp, start=a, confirmed=True, since=since)
    if now - since < p[P.DHW_START_GRACE] * 60:
        return (a, b, kind, cap), dict(state="waiting", setpoint=sp, start=a, confirmed=False, since=since)
    return None, dict(state="not_started", setpoint=sp, start=a, confirmed=False, since=since)


def forced_slots(hb, t0: float, N: int, min_block_hours: float):
    """(bool array of slots in which heating must run, slots the running block already ran) for a running block hb = (start, end):
    forced from t0 until start + minimum block length only (never beyond its planned end)"""
    force = np.zeros(N, bool)
    if not hb:
        return force, 0
    L = max(1, int(round(min_block_hours / DT)))
    until = min(hb[1], hb[0] + L * STEP)
    n = int(math.ceil((until - t0) / STEP - 1e-9))
    if n > 0:
        force[:min(N, n)] = True
    return force, max(0, int(round((t0 - hb[0]) / STEP)))


def run(raw: dict, options: dict, lang: str = "en", previous: dict | None = None) -> dict:
    """one planning run. raw: see inputs.py; options: integration options; previous: last result (running blocks)"""
    p = P.normalise(options)
    tz = ZoneInfo(raw.get("tz") or "UTC")
    now = raw["now"]
    st = raw["states"]
    live = live_battery(p, st)
    live.update(live_dhw_low(p, st))
    t0 = I.q0(now)
    model = Model(p)
    # state estimate from the recent history
    th = np.arange(t0 - HIST_DAYS_STATE * 86400, t0, STEP, dtype=float)
    prof = house_profile(raw, p, tz)
    hinfo = {}
    Tr_h, To_h, Q_h, PV_h, E_h = history(raw, p, th, tz, prof, hinfo)
    cur = I.temp_c(I.state_num(st, p.get(P.ROOM_TEMP_ENTITY)), I.state_unit(st, p.get(P.ROOM_TEMP_ENTITY)))
    if cur is not None and not ROOM_RANGE[0] <= cur <= ROOM_RANGE[1]:
        cur = None
    if np.all(np.isnan(Tr_h)):
        if cur is None:
            raise PlanError("no room temperature history")
        Tr_h[-1] = cur
    U = Model.U(To_h, PV_h, Q_h, E_h)
    x0, state_reset = state_estimate(model, U, Tr_h, cur)
    if hinfo.get("dropped"):
        _LOGGER.debug("implausible history samples left out: %s", hinfo["dropped"])
    if state_reset:
        _LOGGER.warning("implausible state estimate (%s), now [%.2f, %.2f, %.2f]", state_reset, *x0)
    inp, ps, tk_end = build_inputs(raw, p, tz, t0, prof)
    t = inp["t"]
    N = len(t)
    nk = int(inp["known"].sum())
    # ------------------------------------------------------------------ hot water
    mode = p[P.DHW_MODE]
    busy = legionella_busy(raw, p, t)
    dres = None
    dcheck = dict(state=None)
    dsp = dhw_setpoint(p, st)
    dpause = dhw_pause(raw, p, tz) if mode == "plan" else None
    done, last_end, tank = tank_status(raw, p, tz)
    leg = legionella_status(raw, p, tz) if mode == "plan" else None
    if mode == "schedule":
        dhw, dkw = schedule_dhw(p, t, I.blocks_from_text((st.get(p.get(P.DHW_SCHEDULE_ENTITY)) or {}).get("s")))
        dheat = None
    elif mode == "plan":
        running, dcheck = dhw_running(raw, p, previous, tank, dsp, dpause, bool(busy[0]))
        # an external legionella cycle that runs now already does the job: no planned one on top
        leg_req = leg if (leg is not None and not busy[0]) else None
        dres = plan_dhw(p, t, inp["To"], inp["price"], inp["feed"], inp["E"], inp["PV"], inp["peak"], busy, tz, now,
                        done_today=done, last_end=last_end, running=running, known=inp["known"], legionella=leg_req, tank=tank,
                        not_before=dpause)
        dhw, dkw, dheat = dres["dhw"], dres["dhw_kw"], dres["dhw_heat"]
        # the heat pump charges by itself when the tank is below low - hysteresis (up to the low temperature; while the
        # execution is paused up to its live setpoint)
        own = dsp if (dpause is not None and dsp is not None) else p[P.DHW_LOW]
        if tank is not None and tank < own - p[P.DHW_HYST] and not dhw[0]:
            if p[P.DHW_SIZING] == "model":
                r = dhw_charge(p, tank, own, inp["To"])
                n, e, q = min(N, r["slots"]), r["el"], r["heat"]
            else:
                n = min(N, max(1, int(round(p[P.DHW_CHARGE_HOURS] / DT))))
                e = np.full(n, p[P.DHW_POWER])
                q = e * dhw_cop(p, inp["To"][:n])
            dhw[:n] = True
            dheat[:n] = np.where(dkw[:n] >= e[:n], dheat[:n], q[:n])
            dkw[:n] = np.maximum(dkw[:n], e[:n])
    else:
        dhw, dkw = np.zeros(N, bool), np.zeros(N)
        dheat = None
    if dheat is None:
        dheat = dkw * dhw_cop(p, inp["To"])
    lg = busy & ~dhw
    dhw = dhw | busy
    dkw = np.where(lg, p[P.DHW_POWER], dkw)
    dheat = np.where(lg, p[P.DHW_POWER] * dhw_cop(p, inp["To"]), dheat)
    inp["dhw"], inp["dhw_kw"], inp["dhw_heat"] = dhw, dkw, dheat
    # ------------------------------------------------------------------ heating (+ battery)
    joint = p[P.PLAN_BATTERY]
    bat = None
    soc0 = 0.0
    if joint:
        bat = battery_options(p)
        sid = p.get(P.BATTERY_SOC_ENTITY)
        v = I.state_num(st, sid, 0.0) or 0.0
        soc0 = v * bat["cap"] / 100 if (I.state_unit(st, sid) or "").strip() == "%" else v
    # a heating block that already started (in the previous plan) is kept until it reached the minimum block length;
    # after that the optimiser decides freely whether it goes on
    hb = None
    for a, b in ((previous or {}).get("_heat_blocks") or []):
        if a <= now < b and b > t0:
            hb = (float(a), float(b))
    force, r0 = forced_slots(hb, t0, N, p[P.MIN_BLOCK_HOURS])
    tt = time.time()
    o = plan(p, model, inp, x0, r0=r0, battery=bat, soc0=soc0, force=force)
    secs = time.time() - tt
    bl = blocks(o["u"][:nk], t0)
    if hb and bl and bl[0][0] == t0 and hb[0] < t0:   # report the running block from its original start
        bl[0] = (hb[0], bl[0][1])
    res = dict(
        created=_iso(now, tz, "seconds"), net_metering=ps["net_metering"], net_metering_fee=round(ps["fee"], 4),
        battery_planned=joint, runtime_s=round(secs, 2), state_estimate=[round(float(v), 3) for v in x0],
        state_reset=state_reset, history_dropped=hinfo.get("dropped") or None,
        horizon_end=_iso(tk_end, tz), tail_hours=p[P.TAIL_HOURS], min_block_hours=max(1, int(round(p[P.MIN_BLOCK_HOURS] / DT))) * DT,
        schedule=I.blocks_to_text(bl, tz),
        blocks=[[_iso(a, tz), (b - a) / 3600] for a, b in bl],
        blocks_tail=[[_iso(a, tz), (b - a) / 3600] for a, b in blocks(o["u"][nk:], t0 + nk * STEP)],
        heat_kwh=round(float(o["Qd"][:nk].sum() / 4), 1), elec_kwh=round(float(o["Pel"][:nk].sum() / 4), 2),
        cost=round(float(o["cost"][:nk].sum()), 2), tr_min=_r(o["Tr"][1:nk + 1].min()), tr_max=_r(o["Tr"][1:nk + 1].max()),
        t0=_iso(t0, tz), tr=[round(float(v), 2) for v in o["Tr"][:nk + 1]], ts=[round(float(v), 2) for v in o["Ts"][:nk + 1]],
        tr_tail=[round(float(v), 2) for v in o["Tr"][nk + 1:]], ts_tail=[round(float(v), 2) for v in o["Ts"][nk + 1:]],
        price_forecast=inp.get("price_forecast"), post_check=o.get("post_check"),
        running_block=[_iso(bl[0][0], tz), _iso(bl[0][1], tz)] if hb and bl and bl[0][0] <= now < bl[0][1] else None)
    on = np.flatnonzero(np.asarray(o["u"]) > 0)
    if on.size:
        i = int(on[0])
        res.update(next_block_start=_iso(hb[0] if (hb and i == 0) else t0 + i * STEP, tz), tr_next_block=_r(o["Tr"][i]),
                   tr_min_until_next_block=_r(o["Tr"][1:i + 1].min()) if i > 0 else _r(o["Tr"][0]))
    else:
        res.update(next_block_start=None, tr_next_block=None, tr_min_until_next_block=_r(o["Tr"][1:].min()))
    # cost build-up until the end of the known prices: house (incl. hot water) -> + heat pump -> + solar -> + battery
    E_ = inp["E"][:nk] + inp["dhw_kw"][:nk]
    Pel_ = o["Pel"][:nk]
    PV_ = inp["PV"][:nk]
    pr_ = inp["price"][:nk]
    fd_ = inp["feed"][:nk]
    n1 = E_
    n2 = n1 + Pel_
    n3 = n2 - PV_
    n4 = o["net"][:nk]
    c1, c2, c3, c4 = (elec_cost(n, pr_, fd_) for n in (n1, n2, n3, n4))
    wear_ = float(bat["wear"]) * float(np.maximum(-o["bat"][:nk], 0).sum() * DT) if bat else 0.0
    res["cost_breakdown"] = dict(
        house=round(c1, 2), heat_pump=round(c2 - c1, 2), solar=round(c3 - c2, 2), battery=round(c4 - c3, 2),
        battery_wear=round(wear_, 2), total=round(c4, 2),
        solar_self_kwh=round(float(np.minimum(PV_, np.maximum(n2, 0)).sum() * DT), 1),
        solar_export_kwh=round(float(np.maximum(PV_ - np.maximum(n2, 0), 0).sum() * DT), 1))
    imp = np.maximum(n4, 0) * DT
    exp_ = np.maximum(-n4, 0) * DT
    # per-slot loads on the plan grid (t0, 15 min, nk slots like battery code/soc), kW
    res["house_kw"] = [round(float(v), 2) for v in inp["E"][:nk]]
    res["hp_kw"] = [round(float(v), 2) for v in (Pel_ + inp["dhw_kw"][:nk])]
    res["dhw_kw"] = [round(float(v), 2) for v in inp["dhw_kw"][:nk]]
    res["heat_kw"] = [round(float(v), 2) for v in o["Qd"][:nk]]   # planned space heat (thermal kW, excl. hot water)
    res["house_kw_tail"] = [round(float(v), 2) for v in inp["E"][nk:]]
    res["hp_kw_tail"] = [round(float(v), 2) for v in (o["Pel"][nk:] + inp["dhw_kw"][nk:])]
    res["dhw_kw_tail"] = [round(float(v), 2) for v in inp["dhw_kw"][nk:]]
    res["heat_kw_tail"] = [round(float(v), 2) for v in o["Qd"][nk:]]
    dh = np.asarray(inp["dhw_heat"], float)   # planned hot-water heat (thermal kW)
    res["dhw_heat_kw"] = [round(float(v), 2) for v in dh[:nk]]
    res["dhw_heat_kw_tail"] = [round(float(v), 2) for v in dh[nk:]]
    sg = np.asarray(inp["PV"], float) * p[P.SOLAR_GAIN]   # expected solar heat into the house (model input g x PV), kW
    res["solar_gain_kw"] = [round(float(v), 2) for v in sg[:nk]]
    res["solar_gain_kw_tail"] = [round(float(v), 2) for v in sg[nk:]]
    res["house_profile"] = HP.summary(prof, now, tz)
    res["energy"] = dict(
        house_kwh=round(float(E_.sum() * DT), 1), heat_pump_kwh=round(float(Pel_.sum() * DT), 1),
        hot_water_kwh=round(float(inp["dhw_kw"][:nk].sum() * DT), 1), solar_kwh=round(float(PV_.sum() * DT), 1),
        import_kwh=round(float(imp.sum()), 1), export_kwh=round(float(exp_.sum()), 1),
        import_cost=round(float((imp * pr_).sum()), 2), export_revenue=round(float((exp_ * fd_).sum()), 2),
        import_price_avg=round(float((imp * pr_).sum() / imp.sum()), 4) if imp.sum() > 0.01 else None,
        export_price_avg=round(float((exp_ * fd_).sum() / exp_.sum()), 4) if exp_.sum() > 0.01 else None,
        import_price_min=round(float(pr_.min()), 4) if nk else None, import_price_max=round(float(pr_.max()), 4) if nk else None)
    try:   # money of today: realised from the meters until now + plan until midnight (+ tomorrow, provisional)
        res["today"] = A.day_accounting(raw, p, ps, tz, inp, o, bat)
    except Exception as ex:  # noqa: BLE001 - informative only, never blocks the plan
        res["today"] = dict(error=f"{type(ex).__name__}: {ex}")
    alt = None
    if bat is not None:
        try:
            o2 = plan(p, model, inp, x0, r0=r0, battery=None, force=force)
            bl2 = blocks(o2["u"], t0)
            if hb and bl2 and bl2[0][0] == t0 and hb[0] < t0:
                bl2[0] = (hb[0], bl2[0][1])
            alt = [[_iso(a2, tz), (b2 - a2) / 3600] for a2, b2 in bl2]
            # expected battery benefit: electricity cost without battery - with battery (incl. wear), corrected for the
            # change in stored energy (start and end valued at the optimiser's terminal value per kWh) and for the
            # difference in stored heat at the end (heating may shift between the two plans)
            pr_all, fd_all = np.asarray(inp["price"], float), np.asarray(inp["feed"], float)
            vb_ = float(o.get("vb", 0.0))
            vh_ = float(o.get("vh", 0.0))

            def _benefit(n):
                c_b = elec_cost(o["net"][:n], pr_all[:n], fd_all[:n]) + float(bat["wear"]) * float(np.maximum(-o["bat"][:n], 0).sum() * DT)
                c_n = elec_cost(o2["net"][:n], pr_all[:n], fd_all[:n])
                d_soc = float(o["soc"][n]) - float(o["soc"][0])
                d_heat = vh_ * (model.Cr * (float(o["Tr"][n]) - float(o2["Tr"][n])) + model.Cs * (float(o["Ts"][n]) - float(o2["Ts"][n])))
                return c_n - c_b + vb_ * d_soc + d_heat

            N_all = len(o["net"])
            b_all = _benefit(N_all)
            res["battery_benefit"] = dict(
                known_prices=round(_benefit(nk), 2), horizon=round(b_all, 2), hours=round(N_all * DT, 1),
                per_day=round(b_all / (N_all * DT) * 24, 2) if N_all else None, stored_value_per_kwh=round(vb_, 4),
                soc_start_kwh=round(float(o["soc"][0]), 2), soc_end_kwh=round(float(o["soc"][N_all]), 2))
        except Exception as ex:  # informative only
            alt = str(ex)
    res["blocks_without_battery"] = alt
    # ------------------------------------------------------------------ battery output
    if joint:
        code_all = battery_code(o)
        code = code_all[:nk]
        segs = battery_segments(t0, code, o["bat"][:nk], tz)
        t_end = t0 + nk * STEP
        segs_tail = battery_segments(t_end, code_all[nk:], o["bat"][nk:], tz)
        res["battery"] = dict(code=(dt.datetime.fromtimestamp(t0, tz).strftime("%Y-%m-%dT%H:%M") + "|" + code)[:255],
                              soc=[round(float(v), 2) for v in o["soc"][:nk + 1]], blocks=segs,
                              summary=battery_summary(segs, lang), capacity=p[P.BAT_CAPACITY], soc_now=round(soc0, 2),
                              charge_kw=round(p[P.BAT_CHARGE_KW], 3), discharge_kw=round(p[P.BAT_DISCHARGE_KW], 3),
                              live_charge_kw=_r(bat.get("p_ch0"), 3), live_discharge_kw=_r(bat.get("p_dis0"), 3),
                              min_soc=round(p[P.BAT_MIN_SOC], 1), max_soc=round(p[P.BAT_MAX_SOC], 1), live_limits=live,
                              prices=[round(float(v), 4) for v in inp["price"][:nk]],
                              feed_in_prices=[round(float(v), 4) for v in inp["feed"][:nk]],
                              # tentative plan after horizon_end (expected prices): slot i starts at horizon_end + i x 15 min,
                              # soc_tail[i] = charge at the end of that slot (soc[-1] is the charge at horizon_end)
                              horizon_end=_iso(t_end, tz), code_tail=code_all[nk:],
                              soc_tail=[round(float(v), 2) for v in o["soc"][nk + 1:]], blocks_tail=segs_tail,
                              prices_tail=[round(float(v), 4) for v in inp["price"][nk:]],
                              feed_in_prices_tail=[round(float(v), 4) for v in inp["feed"][nk:]])
    # ------------------------------------------------------------------ hot water output
    dblocks = dres["blocks"] if dres else (I.blocks_from_text((st.get(p.get(P.DHW_SCHEDULE_ENTITY)) or {}).get("s"))
                                            if mode == "schedule" else [])
    det = []
    if dres:
        def _sp(kind):
            return p[P.LEGIONELLA_SETPOINT] if kind == "legionella" else (p[P.DHW_HOT_TARGET] if kind == "hot" else None)
        for d in dres["details"]:
            kind = d.get("kind", "normal")
            det.append(dict(start=_iso(d["start"], tz), end=_iso(d["end"], tz), kwh=round(d["kwh"], 2),
                            cost=round(d["cost"], 2), price=round(d["price"], 4), provisional=d["provisional"],
                            kind=kind, setpoint=_sp(kind), heat_kwh=_r(d.get("heat_kwh"), 2), start_temp=_r(d.get("start_temp"), 1)))
        for a, b in dres["fixed"]:
            kind = dres["kinds"].get(float(a), "normal")
            det.insert(0, dict(start=_iso(a, tz), end=_iso(b, tz), kwh=None, cost=None, price=None, provisional=False, running=True,
                               kind=kind, setpoint=_sp(kind), confirmed=bool(dcheck.get("confirmed"))))
    nxt = next((a for a, b in sorted(dblocks) if b > now), None)
    res["hot_water"] = dict(mode=mode, blocks=det if dres else [dict(start=_iso(a, tz), end=_iso(b, tz)) for a, b in dblocks if b > now],
                            schedule=I.blocks_to_text([b for b in dblocks if b[1] > now], tz)[:255],
                            next_start=_iso(nxt, tz) if nxt else None, done_today=bool(done),
                            tank_temp=_r(tank, 1), legionella_active=bool(busy[0]),
                            kwh=round(float(dkw[:nk].sum() * DT), 1),
                            summary=dhw_summary(det if dres else [dict(start=_iso(a, tz), end=_iso(b, tz)) for a, b in dblocks if b > now],
                                                mode, lang, tz),
                            legionella=_legionella_attr(leg, dres, p, tz),
                            paused_until=_iso(dpause, tz) if dpause else None,
                            start_check=dict(state=dcheck.get("state"), setpoint=_r(dsp, 1),
                                             since=_iso(dcheck["since"], tz) if dcheck.get("since") else None))
    res["_dhw_blocks"] = _dhw_block_state(dres, dcheck, now)
    res["_heat_blocks"] = [list(b) for b in bl]
    res["settings"] = {k: p[k] for k in SETTINGS_SHOWN}
    return res


def _dhw_block_state(dres: dict | None, dcheck: dict, now: float) -> list:
    """[start, end, kind, confirmed, due since] per planned charge for the next run (see dhw_running). A charge that was
    not executed and is planned again right away keeps the time since which it is due, so it is not given a new grace
    period on every run."""
    out = []
    fixed = {float(x[0]) for x in (dres["fixed"] if dres else [])}
    for a, b in (dres["blocks"] if dres else []):
        kind = dres["kinds"].get(float(a), "normal")
        confirmed, since = False, None
        if float(a) in fixed:
            confirmed, since = bool(dcheck.get("confirmed")), dcheck.get("since")
        elif dcheck.get("state") == "not_started" and a <= now:
            since = dcheck.get("since")
        out.append([a, b, kind, confirmed, since])
    return out


def _legionella_attr(leg: dict | None, dres: dict | None, p: dict, tz: ZoneInfo) -> dict:
    if leg is None:
        return dict(mode=p[P.LEGIONELLA_MODE])
    li = (dres or {}).get("legionella") or {}
    return dict(mode="plan", last=_iso(leg["last"], tz) if leg["last"] is not None else None, source=leg["source"],
                earliest=_iso(leg["earliest"], tz), due=_iso(leg["due"], tz), overdue=bool(leg["overdue"]),
                temp=p[P.LEGIONELLA_TEMP], setpoint=p[P.LEGIONELLA_SETPOINT],
                planned_start=_iso(li["start"], tz) if li.get("start") is not None else None,
                planned_end=_iso(li["end"], tz) if li.get("end") is not None else None,
                running=bool(li.get("running")), note=li.get("reason"))


# ------------------------------------------------------------------------------------------------ readable summaries
_T = {
    "en": dict(charge="Charge", discharge="Discharge", solar="solar", grid="grid", house="house", export="export",
               nothing="nothing planned", off="off", none="no charge planned", provisional="provisional", running="running",
               legionella="legionella", hot="hot"),
    "nl": dict(charge="Laden", discharge="Ontladen", solar="zon", grid="net", house="huis", export="export",
               nothing="niets gepland", off="uit", none="geen lading gepland", provisional="voorlopig", running="loopt",
               legionella="legionella", hot="heet"),
}
_SRC = {"C": "solar", "F": "grid", "D": "house", "X": "export"}


def _num(x: float, lang: str) -> str:
    s = f"{x:g}"
    return s.replace(".", ",") if lang == "nl" else s


def battery_code(o: dict) -> str:
    """one letter per slot: C charge from solar, F charge from the grid, D discharge for the house, X with export, P idle"""
    net0 = o["net"] - o["bat"]
    code = ""
    for k in range(len(o["bat"])):
        b = o["bat"][k]
        if b > 0.05:
            code += "C" if net0[k] < -0.05 and b * DT <= -net0[k] * DT + 0.2 else "F"
        elif b < -0.05:
            code += "D" if o["net"][k] >= -0.05 else "X"
        else:
            code += "P"
    return code


def battery_segments(t0: float, codes: str, bat, tz: ZoneInfo) -> list[dict]:
    """consecutive quarters with the same action (C/F charge, D/X discharge); any pause quarter ends a segment"""
    segs = []
    for i, c in enumerate(codes):
        k = "charge" if c in "CF" else ("discharge" if c in "DX" else None)
        if k is None:
            continue
        if segs and segs[-1]["action"] == k and i == segs[-1]["_end"]:
            g = segs[-1]
        else:
            g = dict(action=k, _start=i, _end=i, kwh=0.0, codes=set())
            segs.append(g)
        g["_end"] = i + 1
        g["kwh"] += abs(float(bat[i])) * DT
        g["codes"].add(c)
    return [dict(action=g["action"], start=_iso(t0 + g["_start"] * STEP, tz, "minutes"), end=_iso(t0 + g["_end"] * STEP, tz, "minutes"),
                 kwh=round(g["kwh"], 1), via=[_SRC[c] for c in sorted(g["codes"])]) for g in segs]


def battery_summary(segs: list[dict], lang: str = "en") -> str:
    T = _T.get(lang, _T["en"])
    if not segs:
        return T["nothing"]
    return "; ".join(f"{T[s['action']]} {s['start'][11:16]}-{s['end'][11:16]} {_num(s['kwh'], lang)} kWh ({', '.join(T[v] for v in s['via'])})"
                     for s in segs)[:255]


def dhw_summary(blocks_: list[dict], mode: str, lang: str, tz: ZoneInfo) -> str:
    T = _T.get(lang, _T["en"])
    if mode == "off":
        return T["off"]
    if not blocks_:
        return T["none"]
    out = []
    for b in blocks_:
        a = dt.datetime.fromisoformat(b["start"])
        e = dt.datetime.fromisoformat(b["end"])
        s = f"{T['charge']} {a.strftime('%d-%m %H:%M')}-{e.strftime('%H:%M')}"
        if b.get("kind") in ("legionella", "hot"):
            s += f" {T[b['kind']]} {_num(b['setpoint'], lang)} °C"
        if b.get("kwh") is not None:
            s += f" {_num(round(b['kwh'], 1), lang)} kWh"
        if b.get("running"):
            s += f" ({T['running']})"
        elif b.get("provisional"):
            s += f" ({T['provisional']})"
        out.append(s)
    return "; ".join(out)[:255]
