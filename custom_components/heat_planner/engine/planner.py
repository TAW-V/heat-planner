"""Full planning run: raw data -> inputs -> hot water plan -> heating (+ battery) plan -> result dictionary.

Read-only: the result is only published as sensor attributes. Nothing here talks to devices.
"""
from __future__ import annotations

import datetime as dt
import math
import time
from zoneinfo import ZoneInfo

import numpy as np

from . import accounting as A
from . import inputs as I
from . import params as P
from . import profile as HP
from .dhw import plan_dhw, schedule_dhw
from .model import DT, Model
from .optimizer import battery_options, blocks, elec_cost, plan

STEP = I.STEP
HIST_DAYS_STATE = 2       # days of 5-minute history for the state estimate
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


def history(raw: dict, p: dict, t, tz: ZoneInfo, prof: dict | None = None):
    """15-min history arrays (room, outdoor, space-heating heat, PV, household electricity) over grid t"""
    st = raw["states"]
    s5 = raw.get("stats_5min") or {}
    room_id, out_id = p.get(P.ROOM_TEMP_ENTITY), p.get(P.OUTDOOR_TEMP_ENTITY)
    Tr = I.series_15min(s5.get(room_id), t)
    if I.state_unit(st, room_id) and I.state_unit(st, room_id).upper().endswith("F"):
        Tr = (Tr - 32) / 1.8
    To = I.series_15min(s5.get(out_id), t)
    if I.state_unit(st, out_id) and I.state_unit(st, out_id).upper().endswith("F"):
        To = (To - 32) / 1.8
    hid = p.get(P.HEAT_OUTPUT_ENTITY)
    if hid:
        mask = s5.get(p.get(P.HEATING_ACTIVE_ENTITY)) if p.get(P.HEATING_ACTIVE_ENTITY) else None
        Qy = I.series_15min(s5.get(hid), t, mask_rows=mask if mask is not None else None)
        unit = I.state_unit(st, hid)
        if unit:
            Qy = Qy * I.unit_factor(unit, "power")
        elif np.nanmax(np.nan_to_num(Qy)) > 50:
            Qy = Qy / 1000
    else:
        Qy = np.zeros(len(t))
    pid = p.get(P.PV_POWER_ENTITY)
    PV = np.nan_to_num(I.series_15min(s5.get(pid), t)) * I.unit_factor(I.state_unit(st, pid), "power") if pid else np.zeros(len(t))
    Eh = house_energy(raw, p)
    Ep = HP.series(prof, t, tz) if prof is not None else np.full(len(t), float(np.median(list(Eh.values()))) if Eh else 0.0)
    E = np.array([Eh.get(x // 3600 * 3600, Ep[i]) for i, x in enumerate(t)])
    ok = ~np.isnan(To)
    To = np.interp(np.arange(len(t)), np.flatnonzero(ok), To[ok]) if ok.any() else np.full(len(t), 8.0)
    return Tr, To, np.nan_to_num(Qy), PV, E


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


def run(raw: dict, options: dict, lang: str = "en", previous: dict | None = None) -> dict:
    """one planning run. raw: see inputs.py; options: integration options; previous: last result (running blocks)"""
    p = P.normalise(options)
    tz = ZoneInfo(raw.get("tz") or "UTC")
    now = raw["now"]
    st = raw["states"]
    live = live_battery(p, st)
    t0 = I.q0(now)
    model = Model(p)
    # state estimate from the recent history
    th = np.arange(t0 - HIST_DAYS_STATE * 86400, t0, STEP, dtype=float)
    prof = house_profile(raw, p, tz)
    Tr_h, To_h, Q_h, PV_h, E_h = history(raw, p, th, tz, prof)
    if np.all(np.isnan(Tr_h)):
        cur = I.temp_c(I.state_num(st, p.get(P.ROOM_TEMP_ENTITY)), I.state_unit(st, p.get(P.ROOM_TEMP_ENTITY)))
        if cur is None:
            raise PlanError("no room temperature history")
        Tr_h[-1] = cur
    U = Model.U(To_h, PV_h, Q_h, E_h)
    X = model.filter(U, Tr_h)
    x0 = model.Ad @ X[-1] + model.Bd @ U[-1]
    inp, ps, tk_end = build_inputs(raw, p, tz, t0, prof)
    t = inp["t"]
    N = len(t)
    nk = int(inp["known"].sum())
    # ------------------------------------------------------------------ hot water
    mode = p[P.DHW_MODE]
    busy = legionella_busy(raw, p, t)
    dres = None
    done, last_end, tank = tank_status(raw, p, tz)
    if mode == "schedule":
        dhw, dkw = schedule_dhw(p, t, I.blocks_from_text((st.get(p.get(P.DHW_SCHEDULE_ENTITY)) or {}).get("s")))
    elif mode == "plan":
        running = None
        for b in ((previous or {}).get("_dhw_blocks") or []):
            if b[0] <= now < b[1]:
                running = tuple(b)
        dres = plan_dhw(p, t, inp["To"], inp["price"], inp["feed"], inp["E"], inp["PV"], inp["peak"], busy, tz, now,
                        done_today=done, last_end=last_end, running=running, known=inp["known"])
        dhw, dkw = dres["dhw"], dres["dhw_kw"]
        # the heat pump charges by itself when the tank is below low - hysteresis
        if tank is not None and tank < p[P.DHW_LOW] - p[P.DHW_HYST] and not dhw[0]:
            n = max(1, int(round(p[P.DHW_CHARGE_HOURS] / DT)))
            dhw[:n] = True
            dkw[:n] = np.maximum(dkw[:n], p[P.DHW_POWER])
    else:
        dhw, dkw = np.zeros(N, bool), np.zeros(N)
    lg = busy & ~dhw
    dhw = dhw | busy
    dkw = np.where(lg, p[P.DHW_POWER], dkw)
    inp["dhw"], inp["dhw_kw"] = dhw, dkw
    # ------------------------------------------------------------------ heating (+ battery)
    joint = p[P.PLAN_BATTERY]
    bat = None
    soc0 = 0.0
    if joint:
        bat = battery_options(p)
        sid = p.get(P.BATTERY_SOC_ENTITY)
        v = I.state_num(st, sid, 0.0) or 0.0
        soc0 = v * bat["cap"] / 100 if (I.state_unit(st, sid) or "").strip() == "%" else v
    # a heating block that already started (in the previous plan) runs on until its planned end
    hb = None
    for a, b in ((previous or {}).get("_heat_blocks") or []):
        if a <= now < b and b > t0:
            hb = (float(a), float(b))
    force = np.zeros(N, bool)
    r0 = 0
    if hb:
        force[:min(N, int(math.ceil((hb[1] - t0) / STEP - 1e-9)))] = True
        r0 = max(0, int(round((t0 - hb[0]) / STEP)))
    tt = time.time()
    o = plan(p, model, inp, x0, r0=r0, battery=bat, soc0=soc0, force=force)
    secs = time.time() - tt
    bl = blocks(o["u"][:nk], t0)
    if hb and bl and bl[0][0] == t0 and hb[0] < t0:   # report the running block from its original start
        bl[0] = (hb[0], bl[0][1])
    res = dict(
        created=_iso(now, tz, "seconds"), net_metering=ps["net_metering"], net_metering_fee=round(ps["fee"], 4),
        battery_planned=joint, runtime_s=round(secs, 2), state_estimate=[round(float(v), 3) for v in x0],
        horizon_end=_iso(tk_end, tz), tail_hours=p[P.TAIL_HOURS], min_block_hours=max(1, int(round(p[P.MIN_BLOCK_HOURS] / DT))) * DT,
        schedule=I.blocks_to_text(bl, tz),
        blocks=[[_iso(a, tz), (b - a) / 3600] for a, b in bl],
        blocks_tail=[[_iso(a, tz), (b - a) / 3600] for a, b in blocks(o["u"][nk:], t0 + nk * STEP)],
        heat_kwh=round(float(o["Qd"][:nk].sum() / 4), 1), elec_kwh=round(float(o["Pel"][:nk].sum() / 4), 2),
        cost=round(float(o["cost"][:nk].sum()), 2), tr_min=_r(o["Tr"][1:nk + 1].min()), tr_max=_r(o["Tr"][1:nk + 1].max()),
        t0=_iso(t0, tz), tr=[round(float(v), 2) for v in o["Tr"][:nk + 1]], ts=[round(float(v), 2) for v in o["Ts"][:nk + 1]],
        tr_tail=[round(float(v), 2) for v in o["Tr"][nk + 1:]],
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
    res["house_kw_tail"] = [round(float(v), 2) for v in inp["E"][nk:]]
    res["hp_kw_tail"] = [round(float(v), 2) for v in (o["Pel"][nk:] + inp["dhw_kw"][nk:])]
    res["dhw_kw_tail"] = [round(float(v), 2) for v in inp["dhw_kw"][nk:]]
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
        for d in dres["details"]:
            det.append(dict(start=_iso(d["start"], tz), end=_iso(d["end"], tz), kwh=round(d["kwh"], 2),
                            cost=round(d["cost"], 2), price=round(d["price"], 4), provisional=d["provisional"]))
        for a, b in dres["fixed"]:
            det.insert(0, dict(start=_iso(a, tz), end=_iso(b, tz), kwh=None, cost=None, price=None, provisional=False, running=True))
    nxt = next((a for a, b in sorted(dblocks) if b > now), None)
    res["hot_water"] = dict(mode=mode, blocks=det if dres else [dict(start=_iso(a, tz), end=_iso(b, tz)) for a, b in dblocks if b > now],
                            schedule=I.blocks_to_text([b for b in dblocks if b[1] > now], tz)[:255],
                            next_start=_iso(nxt, tz) if nxt else None, done_today=bool(done),
                            tank_temp=_r(tank, 1), legionella_active=bool(busy[0]),
                            kwh=round(float(dkw[:nk].sum() * DT), 1),
                            summary=dhw_summary(det if dres else [dict(start=_iso(a, tz), end=_iso(b, tz)) for a, b in dblocks if b > now],
                                                mode, lang, tz))
    res["_dhw_blocks"] = [list(b) for b in (dres["blocks"] if dres else [])]
    res["_heat_blocks"] = [list(b) for b in bl]
    res["settings"] = {k: p[k] for k in SETTINGS_SHOWN}
    return res


# ------------------------------------------------------------------------------------------------ readable summaries
_T = {
    "en": dict(charge="Charge", discharge="Discharge", solar="solar", grid="grid", house="house", export="export",
               nothing="nothing planned", off="off", none="no charge planned", provisional="provisional", running="running"),
    "nl": dict(charge="Laden", discharge="Ontladen", solar="zon", grid="net", house="huis", export="export",
               nothing="niets gepland", off="uit", none="geen lading gepland", provisional="voorlopig", running="loopt"),
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
    """consecutive quarters with the same action (C/F charge, D/X discharge); one-quarter pauses are bridged"""
    segs = []
    for i, c in enumerate(codes):
        k = "charge" if c in "CF" else ("discharge" if c in "DX" else None)
        if k is None:
            continue
        if segs and segs[-1]["action"] == k and i - segs[-1]["_end"] <= 2:
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
        if b.get("kwh") is not None:
            s += f" {_num(round(b['kwh'], 1), lang)} kWh"
        if b.get("running"):
            s += f" ({T['running']})"
        elif b.get("provisional"):
            s += f" ({T['provisional']})"
        out.append(s)
    return "; ".join(out)[:255]
