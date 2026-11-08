"""Domestic hot water (tank) planning.

The heat pump cannot heat the house while it charges the tank, so hot water is planned first and the
space-heating/battery optimiser then sees the planned charges as busy slots with a known electric load
(sequential planning; the space-heating DP adapts around them).

A charge takes `dhw_charge_hours` at `dhw_power_kw` electric (at the reference COP, scaled with the outdoor
temperature through the hot-water COP). The planner places `dhw_charges_per_day` charges on every calendar day
of the horizon at the lowest marginal electricity cost (import price, or the missed export revenue when solar
covers it), at least `dhw_min_gap_hours` apart, never inside peak periods or a running legionella cycle.
A charge that already happened today (tank reached target - hysteresis) counts for today.
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
             last_end: float | None = None, running: tuple[float, float] | None = None, known=None) -> dict:
    """returns dict(blocks=[(start, end)], dhw=bool array, dhw_kw=array, details=[...])"""
    t = np.asarray(t, float)
    N = len(t)
    n = max(1, int(round(p[P.DHW_CHARGE_HOURS] / DT)))
    gap = int(round(p[P.DHW_MIN_GAP] / DT))
    per_day = int(p[P.DHW_PER_DAY])
    pw = slot_power(p, To)
    net0 = np.asarray(E, float) - np.asarray(PV, float)
    pr = np.asarray(price, float)
    fd = np.asarray(feed, float)

    def mc(x):
        return pr * np.maximum(x, 0) - fd * np.maximum(-x, 0)

    slot_cost = DT * (mc(net0 + pw) - mc(net0))
    blocked = np.asarray(busy, bool).copy()
    if p[P.AVOID_PEAKS]:
        blocked |= np.asarray(peak, bool)
    dhw = np.zeros(N, bool)
    dkw = np.zeros(N)
    fixed = []
    last = -1e9 if last_end is None else (last_end - t[0]) / STEP
    day_of = [dt.datetime.fromtimestamp(x, tz).date() for x in t]
    today = dt.datetime.fromtimestamp(now, tz).date()
    have = {today: min(done_today, per_day)}
    if running is not None and running[1] > t[0]:
        k1 = int(min(N, np.ceil((running[1] - t[0]) / STEP)))
        dhw[:k1] = True
        dkw[:k1] = pw[:k1]
        fixed.append((running[0], t[0] + k1 * STEP))
        d0 = dt.datetime.fromtimestamp(running[0], tz).date()
        have[d0] = have.get(d0, 0) + 1
        last = max(last, k1)
        blocked[:k1] = True
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
    blocks = list(fixed)
    details = []
    for k in best[1]:
        dhw[k:k + n] = True
        dkw[k:k + n] = pw[k:k + n]
        blocks.append((t[k], t[k] + n * STEP))
        el = float(pw[k:k + n].sum() * DT)
        details.append(dict(start=t[k], end=t[k] + n * STEP, kwh=el, cost=float(cost[k]),
                            price=float((pr[k:k + n] * pw[k:k + n]).sum() / pw[k:k + n].sum()),
                            provisional=bool(known is not None and not np.asarray(known)[k:k + n].all())))
    return dict(blocks=sorted(blocks), dhw=dhw, dhw_kw=dkw, details=details, fixed=fixed)


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
