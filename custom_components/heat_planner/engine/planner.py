"""Full planning run: raw data -> inputs -> hot water plan -> heating (+ battery) plan -> result dictionary.

Read-only: the result is only published as sensor attributes. Nothing here talks to devices.
"""
from __future__ import annotations

import datetime as dt
import time
from zoneinfo import ZoneInfo

import numpy as np

from . import inputs as I
from . import params as P
from .dhw import plan_dhw, schedule_dhw
from .model import DT, Model
from .optimizer import battery_options, blocks, elec_cost, plan

STEP = I.STEP
HIST_DAYS_STATE = 2       # days of 5-minute history for the state estimate
# options echoed in the plan sensor (so dashboards can show the active settings)
SETTINGS_SHOWN = (P.MIN_TEMP, P.MAX_TEMP, P.HARD_MIN_TEMP, P.SAFETY_MARGIN, P.MIN_BLOCK_HOURS, P.START_COST, P.TAIL_HOURS,
                  P.AVOID_PEAKS, P.PLAN_BATTERY, P.DHW_MODE, P.HEAT_LOSS, P.MASS_CAPACITY)


class PlanError(Exception):
    """input missing or unusable"""


def _iso(x: float, tz: ZoneInfo, spec: str = "auto") -> str:
    return dt.datetime.fromtimestamp(x, tz).isoformat(timespec=spec)


def _r(x, n=2):
    return None if x is None else round(float(x), n)


def history(raw: dict, p: dict, t, tz: ZoneInfo):
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
    Em = I.median_by_hour(Eh, raw["now"], tz)
    E = np.array([Eh.get(x // 3600 * 3600, Em[dt.datetime.fromtimestamp(x, tz).hour]) for x in t])
    ok = ~np.isnan(To)
    To = np.interp(np.arange(len(t)), np.flatnonzero(ok), To[ok]) if ok.any() else np.full(len(t), 8.0)
    return Tr, To, np.nan_to_num(Qy), PV, E


def house_energy(raw: dict, p: dict) -> dict:
    st = raw["states"]
    ids = {"import": p.get(P.IMPORT_ENERGY_ENTITY), "export": p.get(P.EXPORT_ENERGY_ENTITY),
           "pv": p.get(P.PV_ENERGY_ENTITY), "heat_pump": p.get(P.HP_ENERGY_ENTITY)}
    scales = {k: I.unit_factor(I.state_unit(st, v), "energy") for k, v in ids.items() if v}
    return I.hourly_house_energy(raw.get("stats_hour") or {}, ids, scales)


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


def build_inputs(raw: dict, p: dict, tz: ZoneInfo, t0: int):
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
    Emed = I.median_by_hour(house_energy(raw, p), now, tz)
    hh = np.array([dt.datetime.fromtimestamp(x, tz).hour for x in t])
    price = np.array([Pd.get(x, np.nan) for x in t])
    known = ~np.isnan(price)
    price[~known] = np.array(Pmed)[hh[~known]]
    E = np.array(Emed)[hh]
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
    ps = price_settings(raw, p)
    feed = I.feed_prices(price, ps["net_metering"], ps["K"], ps["vat"], ps["fee"], p[P.EXPORT_FEE])
    return dict(t=t, hours=(t - t0) / 3600, To=To, PV=PV, E=E, price=price, feed=feed, peak=pk, known=known,
                dhw=np.zeros(N, bool), dhw_kw=np.zeros(N)), ps, tk_end


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
    t0 = I.q0(now)
    model = Model(p)
    # state estimate from the recent history
    th = np.arange(t0 - HIST_DAYS_STATE * 86400, t0, STEP, dtype=float)
    Tr_h, To_h, Q_h, PV_h, E_h = history(raw, p, th, tz)
    if np.all(np.isnan(Tr_h)):
        cur = I.temp_c(I.state_num(st, p.get(P.ROOM_TEMP_ENTITY)), I.state_unit(st, p.get(P.ROOM_TEMP_ENTITY)))
        if cur is None:
            raise PlanError("no room temperature history")
        Tr_h[-1] = cur
    U = Model.U(To_h, PV_h, Q_h, E_h)
    X = model.filter(U, Tr_h)
    x0 = model.Ad @ X[-1] + model.Bd @ U[-1]
    inp, ps, tk_end = build_inputs(raw, p, tz, t0)
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
    tt = time.time()
    o = plan(p, model, inp, x0, battery=bat, soc0=soc0)
    secs = time.time() - tt
    bl = blocks(o["u"][:nk], t0)
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
        tr_tail=[round(float(v), 2) for v in o["Tr"][nk + 1:]])
    on = np.flatnonzero(np.asarray(o["u"]) > 0)
    if on.size:
        i = int(on[0])
        res.update(next_block_start=_iso(t0 + i * STEP, tz), tr_next_block=_r(o["Tr"][i]),
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
    res["energy"] = dict(
        house_kwh=round(float(E_.sum() * DT), 1), heat_pump_kwh=round(float(Pel_.sum() * DT), 1),
        hot_water_kwh=round(float(inp["dhw_kw"][:nk].sum() * DT), 1), solar_kwh=round(float(PV_.sum() * DT), 1),
        import_kwh=round(float(imp.sum()), 1), export_kwh=round(float(exp_.sum()), 1),
        import_cost=round(float((imp * pr_).sum()), 2), export_revenue=round(float((exp_ * fd_).sum()), 2),
        import_price_avg=round(float((imp * pr_).sum() / imp.sum()), 4) if imp.sum() > 0.01 else None,
        export_price_avg=round(float((exp_ * fd_).sum() / exp_.sum()), 4) if exp_.sum() > 0.01 else None,
        import_price_min=round(float(pr_.min()), 4) if nk else None, import_price_max=round(float(pr_.max()), 4) if nk else None)
    alt = None
    if bat is not None:
        try:
            o2 = plan(p, model, inp, x0, battery=None)
            alt = [[_iso(a2, tz), (b2 - a2) / 3600] for a2, b2 in blocks(o2["u"], t0)]
        except Exception as ex:  # informative only
            alt = str(ex)
    res["blocks_without_battery"] = alt
    # ------------------------------------------------------------------ battery output
    if joint:
        net0 = o["net"] - o["bat"]
        code = ""
        for k in range(nk):
            b = o["bat"][k]
            if b > 0.05:
                code += "C" if net0[k] < -0.05 and b * DT <= -net0[k] * DT + 0.2 else "F"
            elif b < -0.05:
                code += "D" if o["net"][k] >= -0.05 else "X"
            else:
                code += "P"
        segs = battery_segments(t0, code, o["bat"][:nk], tz)
        res["battery"] = dict(code=(dt.datetime.fromtimestamp(t0, tz).strftime("%Y-%m-%dT%H:%M") + "|" + code)[:255],
                              soc=[round(float(v), 2) for v in o["soc"][:nk + 1]], blocks=segs,
                              summary=battery_summary(segs, lang), capacity=p[P.BAT_CAPACITY], soc_now=round(soc0, 2),
                              prices=[round(float(v), 4) for v in inp["price"][:nk]],
                              feed_in_prices=[round(float(v), 4) for v in inp["feed"][:nk]])
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
