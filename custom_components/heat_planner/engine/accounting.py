"""Day accounting: what today cost so far (from the meters) and what the rest of the day is expected to cost (plan).

Two views of the same money:
  flows        what really goes over the meter: import kWh x import price, export kWh x export price, net
  attribution  who caused it, built up step by step: house (without boiler) -> + boiler (hot water) -> + heat pump
               heating -> + solar (self-use saving and export revenue) -> + battery. Its total equals the net flow.

Pure Python + numpy, no Home Assistant imports. Positive amounts are costs, negative amounts are revenue/savings.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np

from . import inputs as I
from . import params as P

FIVE = 300
DHW_MIN_KWH_5MIN = 0.02     # heat pump electricity per 5 minutes above which a slot without space heating counts as hot water

FLOW_KEYS = ("import_kwh", "import_cost", "export_kwh", "export_revenue", "net")
KWH_KEYS = ("house_kwh", "boiler_kwh", "heating_kwh", "solar_kwh", "solar_self_kwh", "solar_export_kwh",
            "battery_charge_kwh", "battery_discharge_kwh")
ATTR_KEYS = ("house", "boiler", "heating", "solar_self", "solar_export", "battery", "total")


def _cost(net, price, feed) -> float:
    net = np.asarray(net, float)
    return float(np.sum(price * np.maximum(net, 0) - feed * np.maximum(-net, 0)))


def breakdown(house, boiler, heating, pv, price, feed, battery=None, imp=None, exp=None, wear_rate: float = 0.0) -> dict:
    """energy arrays in kWh per slot (house/boiler/heating/pv >= 0; battery = kWh into the battery at the AC side,
    negative when discharging); price/feed in currency per kWh per slot. imp/exp: measured meter energy per slot
    (realised); without them the meter flows follow from the net per slot (plan)."""
    house, boiler, heating, pv = (np.asarray(x, float) for x in (house, boiler, heating, pv))
    price, feed = np.asarray(price, float), np.asarray(feed, float)
    bat = np.zeros(len(house)) if battery is None else np.asarray(battery, float)
    n1 = house
    n2 = n1 + boiler
    n3 = n2 + heating
    n4 = n3 - pv
    n5 = n4 + bat
    if imp is None or exp is None:
        imp, exp = np.maximum(n5, 0), np.maximum(-n5, 0)
    imp, exp = np.asarray(imp, float), np.asarray(exp, float)
    ic = float(np.sum(imp * price))
    er = float(np.sum(exp * feed))
    net = ic - er
    c1, c2, c3, c4 = (_cost(n, price, feed) for n in (n1, n2, n3, n4))
    load = np.maximum(n3, 0)
    self_ = np.minimum(pv, load)
    expo = pv - self_
    s_self = -float(np.sum(self_ * price))
    s_exp = c4 - c3 - s_self          # = - export revenue of the solar surplus (exact by construction)
    out = dict(
        import_kwh=float(imp.sum()), import_cost=ic, export_kwh=float(exp.sum()), export_revenue=er, net=net,
        house_kwh=float(house.sum()), boiler_kwh=float(boiler.sum()), heating_kwh=float(heating.sum()),
        solar_kwh=float(pv.sum()), solar_self_kwh=float(self_.sum()), solar_export_kwh=float(expo.sum()),
        battery_charge_kwh=float(np.maximum(bat, 0).sum()), battery_discharge_kwh=float(np.maximum(-bat, 0).sum()),
        house=c1, boiler=c2 - c1, heating=c3 - c2, solar_self=s_self, solar_export=s_exp,
        battery=net - c4,                 # with measured flows this also absorbs meter/timing differences
        total=net)
    out["battery_wear"] = wear_rate * out["battery_discharge_kwh"]
    return out


def combine(*parts) -> dict:
    parts = [x for x in parts if x]
    if not parts:
        return {}
    out = {k: float(sum(x.get(k, 0.0) for x in parts)) for k in set().union(*parts) if isinstance(parts[0].get(k, 0.0), (int, float))}
    return out


def finish(d: dict) -> dict:
    """round and add average prices"""
    if not d:
        return d
    o = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}
    o["import_price_avg"] = round(d["import_cost"] / d["import_kwh"], 4) if d.get("import_kwh", 0) > 0.05 else None
    o["export_price_avg"] = round(d["export_revenue"] / d["export_kwh"], 4) if d.get("export_kwh", 0) > 0.05 else None
    for k in KWH_KEYS + ("import_kwh", "export_kwh"):
        if k in d:
            o[k] = round(d[k], 1)
    return o


def _changes(rows, scale: float) -> dict[float, float]:
    return {r["start"] / 1000: (I.num(r.get("change"), 0.0) or 0.0) * scale for r in rows or []}


def realised(raw: dict, p: dict, ps: dict, tz: ZoneInfo, until: float) -> dict | None:
    """today from local midnight until `until` (unix, normally the start of the current quarter) from the meters.
    raw["stats_today"]: 5-minute statistics since midnight (change for the energy counters, mean for the battery)."""
    rows = raw.get("stats_today") or {}
    st = raw["states"]
    iid, eid = p.get(P.IMPORT_ENERGY_ENTITY), p.get(P.EXPORT_ENERGY_ENTITY)
    if not iid or iid not in rows:
        return None
    now = raw["now"]
    midnight = dt.datetime.fromtimestamp(now, tz).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    def ch(key):
        e = p.get(key)
        return _changes(rows.get(e), I.unit_factor(I.state_unit(st, e), "energy")) if e else {}

    imp_d, exp_d, pv_d, hp_d = ch(P.IMPORT_ENERGY_ENTITY), ch(P.EXPORT_ENERGY_ENTITY), ch(P.PV_ENERGY_ENTITY), ch(P.HP_ENERGY_ENTITY)
    slots = sorted(s for s in imp_d if midnight <= s and s + FIVE <= until + 1)
    if not slots:
        return None
    # space heating active per 5 minutes (hot water = heat pump electricity while space heating is off)
    ha = {}
    hid = p.get(P.HEATING_ACTIVE_ENTITY)
    for r in ((raw.get("stats_5min") or {}).get(hid) or []) if hid else []:
        v = I.num(r.get("max", r.get("mean")))
        if v is not None:
            ha[r["start"] / 1000] = v > 0
    # battery energy per slot from the state of charge (kWh, or % of the capacity)
    sid = p.get(P.BATTERY_SOC_ENTITY)
    soc = {}
    if sid and sid in rows:
        f = p[P.BAT_CAPACITY] / 100 if (I.state_unit(st, sid) or "").strip() == "%" else I.unit_factor(I.state_unit(st, sid), "energy")
        soc = {r["start"] / 1000: I.num(r.get("mean")) * f for r in rows[sid] if I.num(r.get("mean")) is not None}
    Pd = I.price_dict(raw.get("prices"))
    imp, exp, pv, hp, dhw, bat, pr, fd = ([] for _ in range(8))
    for s in slots:
        imp.append(imp_d.get(s, 0.0))
        exp.append(exp_d.get(s, 0.0) if eid else 0.0)
        pv.append(pv_d.get(s, 0.0))
        h = max(hp_d.get(s, 0.0), 0.0)
        hp.append(h)
        dhw.append(h if (hid and not ha.get(s, False) and h > DHW_MIN_KWH_5MIN) else 0.0)
        b = 0.0
        if s in soc and (s - FIVE) in soc:
            de = soc[s] - soc[s - FIVE]
            b = de / p[P.BAT_CHARGE_EFF] if de > 0 else de * p[P.BAT_DISCHARGE_EFF]
        bat.append(b)
        pr.append(Pd.get(float(I.q0(s)), np.nan))
    pr = np.array(pr, float)
    if np.isnan(pr).all():
        return None
    pr = np.where(np.isnan(pr), np.nanmean(pr), pr)
    fd = I.feed_prices(pr, ps["net_metering"], ps["K"], ps["vat"], ps["fee"], p[P.EXPORT_FEE])
    imp, exp, pv, hp, dhw, bat = (np.array(x, float) for x in (imp, exp, pv, hp, dhw, bat))
    heat = hp - dhw
    house = imp - exp + pv - hp - bat          # residual: everything else behind the meter
    out = breakdown(house, dhw, heat, pv, pr, fd, battery=bat, imp=imp, exp=exp, wear_rate=p[P.BAT_WEAR])
    out["_until"] = slots[-1] + FIVE
    return out


def plan_part(inp: dict, o: dict, sl: slice, bat_opts: dict | None) -> dict | None:
    """plan slots `sl` (15-minute arrays in kW) -> breakdown"""
    E = np.asarray(inp["E"], float)[sl]
    if not len(E):
        return None
    DTH = 0.25
    dhw = np.asarray(inp["dhw_kw"], float)[sl] * DTH
    heat = np.asarray(o["Pel"], float)[sl] * DTH
    pv = np.asarray(inp["PV"], float)[sl] * DTH
    bat = np.asarray(o["bat"], float)[sl] * DTH if "bat" in o else None
    net = np.asarray(o["net"], float)[sl] * DTH
    wear = float(bat_opts["wear"]) if bat_opts else 0.0
    out = breakdown(E * DTH, dhw, heat, pv, np.asarray(inp["price"], float)[sl], np.asarray(inp["feed"], float)[sl],
                    battery=bat, imp=np.maximum(net, 0), exp=np.maximum(-net, 0), wear_rate=wear)
    return out


def day_accounting(raw: dict, p: dict, ps: dict, tz: ZoneInfo, inp: dict, o: dict, bat_opts: dict | None) -> dict:
    """attribute `today` = {realised, rest, total, tomorrow} (each a finished breakdown) + period limits"""
    t = np.asarray(inp["t"], float)
    t0 = float(t[0])
    now = raw["now"]
    d0 = dt.datetime.fromtimestamp(now, tz).replace(hour=0, minute=0, second=0, microsecond=0)
    m1 = (d0 + dt.timedelta(days=1)).replace(tzinfo=None)
    mid1 = dt.datetime(m1.year, m1.month, m1.day, tzinfo=tz).timestamp()
    m2 = (d0 + dt.timedelta(days=2)).replace(tzinfo=None)
    mid2 = dt.datetime(m2.year, m2.month, m2.day, tzinfo=tz).timestamp()
    nd = int(np.searchsorted(t, mid1))
    ne = int(np.searchsorted(t, mid2))
    try:
        rl = realised(raw, p, ps, tz, t0)
    except Exception:  # noqa: BLE001 - accounting must never break planning
        rl = None
    rest = plan_part(inp, o, slice(0, nd), bat_opts)
    tom = plan_part(inp, o, slice(nd, ne), bat_opts) if ne > nd else None
    known = np.asarray(inp["known"], bool)
    res = dict(
        realised=finish({k: v for k, v in (rl or {}).items() if not k.startswith("_")}) or None,
        realised_until=dt.datetime.fromtimestamp(rl["_until"], tz).isoformat(timespec="minutes") if rl else None,
        rest=finish(rest) if rest else None, rest_from=dt.datetime.fromtimestamp(t0, tz).isoformat(timespec="minutes"),
        total=finish(combine(rl and {k: v for k, v in rl.items() if not k.startswith("_")}, rest)) if (rl or rest) else None,
        tomorrow=finish(tom) if tom else None,
        tomorrow_known_share=round(float(known[nd:ne].mean()), 2) if ne > nd else None,
        tomorrow_hours=round((min(ne, len(t)) - nd) * 0.25, 2) if ne > nd else 0.0)
    return res
