"""Household load profile per weekday and quarter hour.

Household load = grid import - grid export + PV - heat pump - (battery charge - battery discharge), so the heat pump
(heating and hot water) and the battery are left out; the optimiser adds the planned heat pump and battery itself.

The profile is built from two sources:
  1. hourly level per weekday and hour from the long-term (hourly) statistics of the last `WEEKS` weeks:
     a recency-weighted trimmed mean per weekday-hour, shrunk towards the all-days value of that hour when there are few
     samples (weight n / (n + K_LEVEL));
  2. the shape of the four quarters inside each hour from 5-minute statistics. Those are purged by the recorder after
     ~10 days, so complete days are kept as quarter values in an archive (stored by the integration). The shape is a
     ratio of sums per quarter (quarter energy / hour energy) over the archived days, per weekday shrunk towards the
     all-days shape (weight n / (n + K_SHAPE)), the all-days shape shrunk towards flat with few days, then normalised so
     the four quarters of an hour keep the hourly energy.
profile[weekday][quarter] = level[weekday][hour] x shape[weekday][quarter], kW, local wall-clock time. No smoothing:
the evening peak keeps its shape.
The constants were chosen on a 28-day out-of-sample backtest of a real house: weekday levels are noisy, so they are shrunk
strongly towards the all-days value; the quarter shape only gets weight as the archive grows (with ~10 days it barely
predicts better than flat).

Pure Python + numpy, no Home Assistant imports.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np

WEEKS = 8               # weeks of hourly statistics for the level per weekday-hour
HALF_LIFE_WEEKS = 2.0   # recency weight half-life (weeks)
TRIM = 0.25             # weighted trimmed mean: share of the weight dropped at each end
K_LEVEL = 10.0          # weekday vs all-days shrinkage of the hourly level (samples)
K_SHAPE = 8.0           # weekday vs all-days shrinkage of the quarter shape (days)
K_FLAT = 14.0           # all-days shape vs flat (days)
ARCHIVE_DAYS = 84       # days of quarter values kept in the archive
MAX_KW = 50.0           # household power above this (per period) is treated as a counter glitch
QPD = 96                # quarters per day
DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

REQUIRED_ROLES = ("import", "export", "pv", "heat_pump")   # must be present for a period when configured
OPTIONAL_ROLES = ("battery_charge", "battery_discharge")   # missing (no statistics yet / not configured) = 0
SIGNS = {"import": 1.0, "export": -1.0, "pv": 1.0, "heat_pump": -1.0, "battery_charge": -1.0, "battery_discharge": 1.0}


def house_changes(stats: dict, ids: dict, scales: dict, period_s: float = 3600, drop_first: bool = False) -> dict[float, float]:
    """household energy per statistics period (kWh) from counter `change` rows; ids/scales keyed by role.
    drop_first: skip the oldest row of every counter (the recorder reports the whole counter as the change of the oldest
    5-minute row once its predecessor is purged). Periods with a missing required counter are skipped, negative values are
    clipped to 0, implausible ones (> MAX_KW) dropped."""
    ch = {}
    for role, eid in ids.items():
        if not eid:
            continue
        rows = [x for x in (stats.get(eid) or []) if x.get("change") is not None]
        if drop_first and rows:
            first = min(x["start"] for x in rows)
            rows = [x for x in rows if x["start"] != first]
        ch[role] = {x["start"] / 1000: float(x["change"]) * scales.get(role, 1.0) for x in rows}
    if "import" not in ch:
        return {}
    req = [r for r in REQUIRED_ROLES if r in ch]
    opt = [r for r in OPTIONAL_ROLES if r in ch]
    out = {}
    lim = MAX_KW * period_s / 3600
    for s in ch["import"]:
        try:
            e = sum(SIGNS[r] * ch[r][s] for r in req)
        except KeyError:
            continue
        e += sum(SIGNS[r] * ch[r].get(s, 0.0) for r in opt)
        if e > lim:
            continue
        out[s] = max(e, 0.0)
    return out


def _wtrim(v, w, trim: float | None = None) -> float | None:
    """weighted trimmed mean (drops `trim` (default TRIM) of the total weight at each end; plain weighted mean below
    5 samples)"""
    trim = TRIM if trim is None else trim
    if not len(v):
        return None
    v, w = np.asarray(v, float), np.asarray(w, float)
    if len(v) < 5 or trim <= 0:
        return float(np.sum(v * w) / np.sum(w))
    o = np.argsort(v, kind="stable")
    v, w = v[o], w[o]
    tot = w.sum()
    lo, hi = trim * tot, (1 - trim) * tot
    c1 = np.cumsum(w)
    c0 = c1 - w
    keep = np.clip(np.minimum(c1, hi) - np.maximum(c0, lo), 0, None)   # part of each weight inside [lo, hi]
    return float(np.sum(v * keep) / np.sum(keep))


def _local(t: float, tz: ZoneInfo) -> dt.datetime:
    return dt.datetime.fromtimestamp(t, tz)


def _midnight(now: float, tz: ZoneInfo) -> float:
    d = _local(now, tz)
    return dt.datetime(d.year, d.month, d.day, tzinfo=tz).timestamp()


def quarter_days(five: dict[float, float], tz: ZoneInfo, before: float) -> dict[str, list[float]]:
    """complete local days before `before` from 5-minute household kWh -> {iso date: [96 kW]} (wall-clock quarters;
    a quarter needs at least 2 of its 3 five-minute values, else the day is incomplete)"""
    acc: dict[str, list[list[float]]] = {}
    for s, e in five.items():
        if s >= before:
            continue
        d = _local(s, tz)
        acc.setdefault(d.date().isoformat(), [[] for _ in range(QPD)])[d.hour * 4 + d.minute // 15].append(e)
    out = {}
    for day, qs in acc.items():
        if all(len(x) >= 2 for x in qs):
            out[day] = [round(float(np.mean(x)) * 12, 3) for x in qs]
    return out


def merge_archive(archive: dict | None, days: dict[str, list[float]], now: float, tz: ZoneInfo) -> dict[str, list[float]]:
    """add complete days to the archive and drop days older than ARCHIVE_DAYS"""
    out = {k: v for k, v in (archive or {}).items() if isinstance(v, list) and len(v) == QPD}
    out.update(days)
    cut = (_local(now, tz).date() - dt.timedelta(days=ARCHIVE_DAYS)).isoformat()
    return {k: out[k] for k in sorted(out) if k >= cut}


def _shape(days: list[list[float]]) -> np.ndarray:
    """quarter / hour-mean, ratio of sums over the days -> [96]"""
    if not days:
        return np.ones(QPD)
    a = np.asarray(days, float)
    hm = a.reshape(len(a), 24, 4).mean(axis=2)
    num = a.sum(axis=0)
    den = np.repeat(hm.sum(axis=0), 4)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)


def _normalise(sh: np.ndarray) -> np.ndarray:
    m = np.repeat(sh.reshape(24, 4).mean(axis=1), 4)
    return np.where(m > 0, sh / np.where(m > 0, m, 1.0), 1.0)


def build(hourly: dict[float, float], archive: dict[str, list[float]] | None, now: float, tz: ZoneInfo) -> dict:
    """profile from hourly household kWh {hour start: kWh} and quarter days {iso date: [96 kW]}; only days before today"""
    today0 = _midnight(now, tz)
    since = today0 - WEEKS * 7 * 86400
    per: dict[tuple[int, int], tuple[list, list]] = {}
    allh: dict[int, tuple[list, list]] = {h: ([], []) for h in range(24)}
    for s, e in hourly.items():
        if e is None or not since <= s < today0:
            continue
        d = _local(s, tz)
        w = 0.5 ** ((today0 - s) / (7 * 86400) / HALF_LIFE_WEEKS)
        vw = per.setdefault((d.weekday(), d.hour), ([], []))
        vw[0].append(e)
        vw[1].append(w)
        allh[d.hour][0].append(e)
        allh[d.hour][1].append(w)
    every = [x for h in range(24) for x in allh[h][0]]
    fallback = float(np.median(every)) if every else 0.0
    level = np.zeros((7, 24))
    n_level = [0] * 7
    for h in range(24):
        a = _wtrim(*allh[h])
        a = fallback if a is None else a
        for wd in range(7):
            v, w = per.get((wd, h), ([], []))
            m = _wtrim(v, w)
            n = len(v)
            level[wd, h] = a if m is None else (n * m + K_LEVEL * a) / (n + K_LEVEL)
            n_level[wd] = max(n_level[wd], n)
    arch = {k: v for k, v in (archive or {}).items() if k < _local(today0, tz).date().isoformat()}
    alld = list(arch.values())
    sh_all = _shape(alld)
    f = len(alld) / (len(alld) + K_FLAT)
    sh_all = f * sh_all + (1 - f)
    days = [[] for _ in range(7)]
    for k, v in arch.items():
        days[dt.date.fromisoformat(k).weekday()].append(v)
    prof = np.zeros((7, QPD))
    for wd in range(7):
        g = len(days[wd]) / (len(days[wd]) + K_SHAPE)
        sh = _normalise(g * _shape(days[wd]) + (1 - g) * sh_all) if days[wd] else _normalise(sh_all)
        prof[wd] = np.repeat(level[wd], 4) * sh
    return dict(days=[[round(float(x), 3) for x in prof[wd]] for wd in range(7)], level_samples=n_level,
                shape_days=[len(x) for x in days], archive_days=len(alld), weeks=WEEKS,
                built=_local(now, tz).isoformat(timespec="seconds"), built_for=_local(today0, tz).date().isoformat())


def valid(profile) -> bool:
    """7 x 96 values built from at least some statistics"""
    try:
        return (len(profile["days"]) == 7 and all(len(d) == QPD for d in profile["days"])
                and max(profile.get("level_samples") or [1]) > 0)
    except (KeyError, TypeError, ValueError):
        return False


def series(profile: dict, t, tz: ZoneInfo) -> np.ndarray:
    """kW per slot start t (local weekday and wall-clock quarter)"""
    days = profile["days"]
    out = np.empty(len(t))
    for i, x in enumerate(t):
        d = _local(float(x), tz)
        out[i] = days[d.weekday()][d.hour * 4 + d.minute // 15]
    return out


def summary(profile: dict, now: float, tz: ZoneInfo) -> dict:
    """compact view for the plan sensor: today per quarter, the weekday profiles (kW per quarter, local time), evening peaks"""
    d0 = _local(now, tz).date()
    days = profile["days"]
    peaks = {}
    for wd in range(7):
        ev = days[wd][64:88]                       # 16:00-22:00
        i = int(np.argmax(ev))
        peaks[DAY_NAMES[wd]] = dict(time=f"{(64 + i) // 4:02d}:{(64 + i) % 4 * 15:02d}", kw=round(ev[i], 2))
    return dict(
        today=[round(x, 2) for x in days[d0.weekday()]],
        weekdays={DAY_NAMES[wd]: [round(x, 2) for x in days[wd]] for wd in range(7)},
        kwh_per_day={DAY_NAMES[wd]: round(sum(days[wd]) / 4, 1) for wd in range(7)},
        evening_peak=peaks, level_samples=profile.get("level_samples"), shape_days=profile.get("shape_days"),
        archive_days=profile.get("archive_days"), weeks=profile.get("weeks"), built=profile.get("built"),
        source=profile.get("source", "statistics"))
