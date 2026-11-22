"""Domestic hot water (tank) planning.

The heat pump cannot heat the house while it charges the tank, so hot water is planned first and the
space-heating/battery optimiser then sees the planned charges as busy slots with a known electric load
(sequential planning; the space-heating DP adapts around them).

A charge takes `dhw_charge_hours` at `dhw_power_kw` electric (at the reference COP, scaled with the outdoor
temperature through the hot-water COP). The planner places `dhw_charges_per_day` charges on every calendar day
of the horizon at the lowest marginal electricity cost (import price, or the missed export revenue when solar
covers it), at least `dhw_min_gap_hours` apart, never inside peak periods or a running legionella cycle.
A charge that already happened today (tank reached target - hysteresis) counts for today.

Legionella (option legionella_mode = plan): the tank must reach `legionella_temp` at least every `legionella_interval_days`.
The deadline is counted from the last time the tank was that hot (restart-proof: from an entity and/or long-term
statistics, see planner.legionella_status). Once the deadline lies inside the horizon (or has passed), one legionella
charge (`legionella_charge_hours` at `legionella_power_kw`) is placed at the lowest incremental cost between
last + `legionella_earliest_days` and the deadline (if overdue: within the next 24 h). It replaces the normal charge of
its calendar day. Optional hot charges: a planned normal charge whose average price is at most `dhw_hot_price_ratio` x
the median import price of the known horizon is published with setpoint `dhw_hot_target_temp` (stores cheap/solar heat).
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np

from . import params as P
from .inputs import STEP
from .model import DT, dhw_cop

SKIP_PENALTY = 25.0   # currency; a day without a possible charge is allowed but strongly avoided


def slot_power(p: dict, To) -> np.ndarray:
    """electric kW during a planned charge (fixed heat per charge, COP depends on the outdoor temperature)"""
    return p[P.DHW_POWER] * p[P.DHW_COP_REF] / dhw_cop(p, To)


def plan_dhw(p: dict, t, To, price, feed, E, PV, peak, busy, tz: ZoneInfo, now: float, done_today: int = 0,
             last_end: float | None = None, running: tuple | None = None, known=None, legionella: dict | None = None) -> dict:
    """returns dict(blocks=[(start, end)], kinds={start: kind}, dhw=bool array, dhw_kw=array, details=[...], legionella=...)

    running: (start, end[, kind]) of a charge that already started; legionella: dict(earliest=ts, due=ts) when a
    legionella charge has to be planned (legionella_mode plan), else None"""
    t = np.asarray(t, float)
    N = len(t)
    n = max(1, int(round(p[P.DHW_CHARGE_HOURS] / DT)))
    nl = max(1, int(round(p[P.LEGIONELLA_CHARGE_HOURS] / DT)))
    gap = int(round(p[P.DHW_MIN_GAP] / DT))
    per_day = int(p[P.DHW_PER_DAY])
    pw = slot_power(p, To)
    pwl = pw * p[P.LEGIONELLA_POWER] / p[P.DHW_POWER]
    net0 = np.asarray(E, float) - np.asarray(PV, float)
    pr = np.asarray(price, float)
    fd = np.asarray(feed, float)

    def mc(x):
        return pr * np.maximum(x, 0) - fd * np.maximum(-x, 0)

    slot_cost = DT * (mc(net0 + pw) - mc(net0))
    slot_cost_l = DT * (mc(net0 + pwl) - mc(net0))
    blocked = np.asarray(busy, bool).copy()
    if p[P.AVOID_PEAKS]:
        blocked |= np.asarray(peak, bool)
    dhw = np.zeros(N, bool)
    dkw = np.zeros(N)
    fixed = []
    kinds: dict[float, str] = {}
    last = -1e9 if last_end is None else (last_end - t[0]) / STEP
    day_of = [dt.datetime.fromtimestamp(x, tz).date() for x in t]
    today = dt.datetime.fromtimestamp(now, tz).date()
    have = {today: min(done_today, per_day)}
    leg_info = None
    planned_leg = None
    if running is not None and running[1] > t[0]:
        rk = running[2] if len(running) > 2 and running[2] else "normal"
        k1 = int(min(N, np.ceil((running[1] - t[0]) / STEP)))
        dhw[:k1] = True
        dkw[:k1] = (pwl if rk in ("legionella", "hot") else pw)[:k1]
        fixed.append((running[0], t[0] + k1 * STEP))
        kinds[float(running[0])] = rk
        d0 = dt.datetime.fromtimestamp(running[0], tz).date()
        have[d0] = have.get(d0, 0) + 1
        last = max(last, k1)
        blocked[:k1] = True
        if rk == "legionella":
            legionella = None          # the running charge is the legionella charge
            leg_info = dict(start=float(running[0]), end=float(t[0] + k1 * STEP), running=True)
    # ------------------------------------------------------------------ legionella charge (before the normal charges)
    if legionella is not None and N > nl:
        due = float(legionella["due"])
        earliest = float(legionella.get("earliest") or t[0])
        horizon_end = t[-1] + STEP
        overdue = due <= now
        lo = max(t[0], earliest)
        hi = (t[0] + 86400) if overdue else due      # latest end of the charge
        if overdue or due <= horizon_end:
            # incremental cost: the legionella charge replaces the cheapest normal charge of its calendar day
            day_best: dict = {}
            for k in range(0, N - n + 1):
                if not blocked[k:k + n].any():
                    c = float(slot_cost[k:k + n].sum())
                    if c < day_best.get(day_of[k], np.inf):
                        day_best[day_of[k]] = c
            best_k, best_c = None, np.inf
            for k in range(0, N - nl + 1):
                if t[k] < lo - 1e-6 or t[k] + nl * STEP > hi + 1e-6 or blocked[k:k + nl].any():
                    continue
                c = float(slot_cost_l[k:k + nl].sum()) - (day_best.get(day_of[k], 0.0) if have.get(day_of[k], 0) < per_day else 0.0)
                if k < last + gap and not overdue:
                    continue
                if c + 1e-7 * k < best_c:
                    best_k, best_c = k, c + 1e-7 * k
            if best_k is not None:
                k = best_k
                dhw[k:k + nl] = True
                dkw[k:k + nl] = pwl[k:k + nl]
                planned_leg = (t[k], t[k] + nl * STEP)
                kinds[float(t[k])] = "legionella"
                have[day_of[k]] = have.get(day_of[k], 0) + 1
                # keep the normal charges of other days at least `gap` away from it
                blocked[max(0, k - gap - n + 1):min(N, k + nl + gap)] = True
                el = float(pwl[k:k + nl].sum() * DT)
                leg_info = dict(start=float(t[k]), end=float(t[k] + nl * STEP), kwh=el, cost=float(slot_cost_l[k:k + nl].sum()),
                                price=float((pr[k:k + nl] * pwl[k:k + nl]).sum() / pwl[k:k + nl].sum()),
                                provisional=bool(known is not None and not np.asarray(known)[k:k + nl].all()))
            else:
                leg_info = dict(start=None, end=None, reason="no free slot before the deadline")
    # candidate starts
    ok = np.zeros(N, bool)
    cost = np.full(N, np.inf)
    for k in range(0, N - n + 1):
        if blocked[k:k + n].any():
            continue
        ok[k] = True
        cost[k] = float(slot_cost[k:k + n].sum())
    # stages: required charges per calendar day, days covered for at least half by the horizon (today always)
    days = []
    for d in sorted(set(day_of)):
        cover = sum(1 for x in day_of if x == d) * DT
        if d == today or cover >= 12:
            days.extend([d] * max(0, per_day - have.get(d, 0)))
    # DP over stages; state = end slot of the last charge
    states = {last: (0.0, [])}
    for d in days:
        cand = [k for k in range(N) if ok[k] and day_of[k] == d]
        new = {}
        for le, (c0, path) in states.items():
            key = le
            v = c0 + SKIP_PENALTY
            if key not in new or v < new[key][0]:
                new[key] = (v, path)
            for k in cand:
                if k < le + gap:
                    continue
                v = c0 + cost[k] + 1e-7 * k
                e = k + n
                if e not in new or v < new[e][0]:
                    new[e] = (v, path + [k])
        states = new
    best = min(states.values(), key=lambda x: x[0]) if states else (0.0, [])
    blocks = list(fixed) + ([planned_leg] if planned_leg else [])
    details = []
    if leg_info and leg_info.get("start") is not None and not leg_info.get("running"):
        details.append(dict(leg_info, kind="legionella"))
    # hot charges: a cheap normal charge is extended to a legionella-length charge at the hot setpoint
    ratio = p[P.DHW_HOT_PRICE_RATIO]
    kn = np.asarray(known, bool) if known is not None else np.ones(N, bool)
    med = float(np.median(pr[kn])) if kn.any() else float(np.median(pr))
    for k in best[1]:
        kind, m = "normal", n
        if ratio > 0 and k + nl <= N and not blocked[k + n:k + nl].any() and not dhw[k + n:k + nl].any():
            avg = float(slot_cost_l[k:k + nl].sum() / max(1e-9, pwl[k:k + nl].sum() * DT))
            if avg <= ratio * med:
                kind, m = "hot", nl
        pk = pwl if kind == "hot" else pw
        dhw[k:k + m] = True
        dkw[k:k + m] = pk[k:k + m]
        blocks.append((t[k], t[k] + m * STEP))
        kinds[float(t[k])] = kind
        el = float(pk[k:k + m].sum() * DT)
        details.append(dict(start=t[k], end=t[k] + m * STEP, kwh=el, cost=float((slot_cost_l if kind == "hot" else slot_cost)[k:k + m].sum()),
                            price=float((pr[k:k + m] * pk[k:k + m]).sum() / pk[k:k + m].sum()),
                            provisional=bool(known is not None and not np.asarray(known)[k:k + m].all()), kind=kind))
    details.sort(key=lambda d: d["start"])
    return dict(blocks=sorted(blocks), kinds=kinds, dhw=dhw, dhw_kw=dkw, details=details, fixed=fixed, legionella=leg_info)


def schedule_dhw(p: dict, t, blocks) -> tuple[np.ndarray, np.ndarray]:
    """busy slots from an external schedule: the first dhw_charge_hours of every block at dhw_power_kw"""
    t = np.asarray(t, float)
    dhw = np.zeros(len(t), bool)
    dkw = np.zeros(len(t))
    for a, b in blocks:
        m = (t >= a) & (t < min(b, a + p[P.DHW_CHARGE_HOURS] * 3600))
        dhw |= m
        dkw[m] = p[P.DHW_POWER]
    return dhw, dkw
