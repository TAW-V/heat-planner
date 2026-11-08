"""Input preparation: turns raw Home Assistant data (states, statistics, service responses) into 15-minute series.

Everything here is pure Python + numpy. The raw data dictionary (built by the coordinator, or by a test) contains:
    now          unix time
    tz           IANA time zone name
    states       {entity_id: {"s": state, "a": attributes, "lc": last_changed unix time}}
    prices       [{"start": iso or unix, "price": currency/kWh}]  (known prices, any slot length)
    weather      hourly forecast list from weather.get_forecasts
    stats_hour   {statistic_id: [{"start": ms, "change"/"mean": value}]}   (8 days, hourly)
    stats_5min   {statistic_id: [{"start": ms, "mean"/"max": value}]}       (2 days, 5-minute)
"""
from __future__ import annotations

import ast
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np

STEP = 900


def ts(s) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    if isinstance(s, dt.datetime):
        return s.timestamp()
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()


def q0(t: float) -> int:
    return int(t // STEP * STEP)


def num(x, default=None):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def as_list(x):
    """attribute values may arrive as python-literal strings (REST API dumps); accept both"""
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return None
    return x


# ---------------------------------------------------------------------------------------------- units
def unit_factor(unit: str | None, kind: str) -> float:
    """multiplier to kW / kWh / currency per kWh"""
    u = (unit or "").strip().lower().replace(" ", "")
    if kind == "power":
        return {"w": 1e-3, "kw": 1.0, "mw": 1e3}.get(u, 1.0)
    if kind == "energy":
        return {"wh": 1e-3, "kwh": 1.0, "mwh": 1e3}.get(u, 1.0)
    if kind == "price":
        if "/mwh" in u:
            return 1e-3
        if u.startswith(("ct", "c/", "cent", "öre", "ore", "øre", "p/")):
            return 1e-2
        return 1.0
    return 1.0


def temp_c(v, unit: str | None):
    if v is None:
        return None
    return (v - 32) / 1.8 if (unit or "").upper().endswith("F") else v


def state_unit(states: dict, eid: str | None) -> str | None:
    return ((states.get(eid) or {}).get("a") or {}).get("unit_of_measurement") if eid else None


def state_num(states: dict, eid: str | None, default=None):
    return num((states.get(eid) or {}).get("s"), default) if eid else default


def state_on(states: dict, eid: str | None):
    """True/False for on/off-like states, None when unknown"""
    s = (states.get(eid) or {}).get("s") if eid else None
    if s in ("on", "true", "True", "home", "open", "heat", "running"):
        return True
    if s in ("off", "false", "False", "not_home", "closed", "idle"):
        return False
    return None


# ---------------------------------------------------------------------------------------------- statistics
def stat_series(rows, key: str) -> dict[float, float]:
    return {r["start"] / 1000: r.get(key) for r in rows or [] if r.get(key) is not None}


def series_15min(rows, t, key="mean", scale=1.0, mask_rows=None):
    """average of 5-minute statistics per 15-min slot of grid t (NaN where no data); with mask_rows, values
    count only where the mask statistic is > 0 (else 0)"""
    n = len(t)
    acc = np.zeros(n)
    c = np.zeros(n)
    md = {x["start"]: x.get("mean") for x in mask_rows} if mask_rows is not None else None
    for x in rows or []:
        v = x.get(key)
        i = int((x["start"] / 1000 - t[0]) // STEP)
        if v is None or not (0 <= i < n):
            continue
        if md is not None:
            v = v if (md.get(x["start"]) or 0) > 0 else 0.0
        acc[i] += v * scale
        c[i] += 1
    out = np.full(n, np.nan)
    out[c > 0] = acc[c > 0] / c[c > 0]
    return out


def hourly_house_energy(stats_hour: dict, ids: dict, scales: dict) -> dict[float, float]:
    """household electricity per hour (kWh) = import - export + PV - heat pump; ids/scales keyed by role"""
    ch = {role: {x["start"] / 1000: x.get("change") for x in stats_hour.get(eid, [])} for role, eid in ids.items() if eid}
    out = {}
    for h in ch.get("import", {}):
        try:
            e = ch["import"][h] * scales.get("import", 1.0)
            if "export" in ch:
                e -= ch["export"][h] * scales.get("export", 1.0)
            if "pv" in ch:
                e += ch["pv"][h] * scales.get("pv", 1.0)
            if "heat_pump" in ch:
                e -= ch["heat_pump"][h] * scales.get("heat_pump", 1.0)
            out[h] = max(e, 0.0)
        except (KeyError, TypeError):
            pass
    return out


def median_by_hour(series: dict, before: float, tz: ZoneInfo, days: int = 7) -> list[float]:
    """same-hour-of-day median over the last `days` days before `before` -> list[24]"""
    b = {h: [] for h in range(24)}
    for t, v in series.items():
        if before - days * 86400 <= t < before and v is not None:
            b[dt.datetime.fromtimestamp(t, tz).hour].append(v)
    allv = [v for l in b.values() for v in l]
    return [float(np.median(b[h])) if b[h] else (float(np.median(allv)) if allv else 0.0) for h in range(24)]


# ---------------------------------------------------------------------------------------------- forecasts
def weather_series(fc, t, tz: ZoneInfo, bias: list[float], unit: str | None = None):
    """hourly forecast temperature interpolated to the slot centres, minus the hour-of-day bias vs the outdoor sensor"""
    tt = np.array([ts(x["datetime"]) for x in fc])
    vv = np.array([temp_c(float(x["temperature"]), unit) for x in fc])
    T = np.interp(np.asarray(t) + STEP / 2, tt + 1800, vv)
    return T - np.array([bias[dt.datetime.fromtimestamp(x, tz).hour] for x in t])


def pv_forecast_series(states: dict, entities, t, scale_tail: float = 0.5):
    """kW per slot from Solcast-style `detailedForecast` attributes (30-min periods, pv_estimate in kW);
    beyond the last forecast day the last day's profile x scale_tail"""
    d = {}
    for e in entities or []:
        for x in as_list(((states.get(e) or {}).get("a") or {}).get("detailedForecast")) or []:
            try:
                d[ts(x["period_start"])] = float(x["pv_estimate"])
            except (KeyError, TypeError, ValueError):
                continue
    if not d:
        return np.zeros(len(t))
    last = max(d)
    out = np.zeros(len(t))
    for i, x in enumerate(t):
        h = x - (x % 1800)
        if h in d:
            out[i] = d[h]
        elif h > last:
            k = h
            while k > last:
                k -= 86400
            out[i] = d.get(k, 0.0) * scale_tail
    return out


# ---------------------------------------------------------------------------------------------- prices
def price_dict(prices) -> dict[float, float]:
    """[{"start", "price"}] -> {slot start: price}; hourly prices are spread over the four quarters"""
    pts = sorted((ts(x["start"]), float(x["price"])) for x in prices or [] if num(x.get("price")) is not None)
    diffs = [b[0] - a[0] for a, b in zip(pts, pts[1:]) if b[0] > a[0]]
    step = min(min(diffs), 3600) if diffs else 3600
    out = {}
    for i, (s, v) in enumerate(pts):
        e = min(s + step, pts[i + 1][0]) if i + 1 < len(pts) and pts[i + 1][0] > s else s + step
        k = q0(s)
        while k < e:
            out[float(k)] = v
            k += STEP
    return out


_START_KEYS = ("start", "start_time", "startsAt", "time", "from", "datetime")
_PRICE_KEYS = ("price", "value", "total", "price_per_kwh", "electricity_price")
_LIST_ATTRS = ("raw_today", "raw_tomorrow", "prices_today", "prices_tomorrow", "prices", "today", "tomorrow", "forecast", "data")


def prices_from_attributes(attrs: dict, unit: str | None) -> list[dict]:
    """generic parser for price entities that publish their price list in attributes
    (e.g. Nord Pool raw_today/raw_tomorrow, ENTSO-e prices, EnergyZero-style lists)"""
    f = unit_factor(unit, "price")
    out = []
    for key in _LIST_ATTRS:
        for x in as_list((attrs or {}).get(key)) or []:
            if not isinstance(x, dict):
                continue
            s = next((x[k] for k in _START_KEYS if k in x), None)
            v = next((x[k] for k in _PRICE_KEYS if k in x and num(x[k]) is not None), None)
            if s is None or v is None:
                continue
            out.append(dict(start=ts(s), price=float(v) * f))
    uniq = {}
    for x in out:
        uniq[x["start"]] = x
    return list(uniq.values())


def feed_prices(p, net_metering: bool, fixed_costs: float, vat: float, fee: float, export_fee: float = 0.0):
    """export price per slot: net metering -> import price - fee; else spot price excl. VAT - export fee"""
    p = np.asarray(p, float)
    return p - fee if net_metering else (p - fixed_costs) / (1 + vat) - export_fee


def peak_periods(attrs: dict) -> list[tuple[float, float]]:
    out = []
    for x in as_list((attrs or {}).get("periods")) or []:
        try:
            out.append((ts(x["start"]), ts(x["end"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def top_share_peaks(t, p, tz: ZoneInfo, share: float):
    """peak flag where no peak periods are known: the top `share` % of each day's slots"""
    flag = np.zeros(len(t), bool)
    if share <= 0:
        return flag
    days = {}
    for i, x in enumerate(t):
        days.setdefault(dt.datetime.fromtimestamp(x, tz).date(), []).append(i)
    for ix in days.values():
        ix = np.array(ix)
        thr = np.percentile(p[ix], 100 - share)
        flag[ix] = p[ix] > thr
    return flag


def blocks_from_text(s) -> list[tuple[float, float]]:
    """'ISO-start/hours;ISO-start/hours' -> [(start, end)]"""
    out = []
    for e in (s or "").split(";"):
        if "/" in e:
            try:
                a = ts(e.split("/")[0])
                out.append((a, a + float(e.split("/")[1]) * 3600))
            except ValueError:
                pass
    return out


def blocks_to_text(bl, tz: ZoneInfo) -> str:
    return ";".join(dt.datetime.fromtimestamp(a, tz).isoformat() + "/" + f"{(b - a) / 3600:g}" for a, b in bl)
