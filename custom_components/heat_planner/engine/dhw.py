"""Domestic hot water (tank) planning.

The heat pump cannot heat the house while it charges the tank, so hot water is planned first and the
space-heating/battery optimiser then sees the planned charges as busy slots with a known electric load
(sequential planning; the space-heating DP adapts around them).

Charge sizing (dhw_sizing = model, default): the length and the electric load of every charge are simulated from the tank
temperature at its start to its target (normal: dhw_target_temp, hot: dhw_hot_target_temp, legionella: legionella_setpoint)
with a heat rate and COP that fall with the tank temperature and rise with the outdoor temperature (model.dhw_charge), plus
`dhw_start_minutes` start-up and `dhw_charge_margin` %, rounded up to 15-minute slots and capped at `dhw_max_charge_hours`.
The start temperature is the current tank temperature (cooled by standing losses) for a charge within the next hours and at
most `dhw_typical_start_temp` for later charges (model.dhw_start_temp). A running charge is extended (never shortened) when
the tank is still further from its target than the remaining planned time covers.
With dhw_sizing = fixed a charge takes `dhw_charge_hours` at `dhw_power_kw` electric (at the reference COP, scaled with the
outdoor temperature through the hot-water COP); legionella and hot charges `legionella_charge_hours` at `legionella_power_kw`.
The planner places `dhw_charges_per_day` charges on every calendar day
of the horizon at the lowest marginal electricity cost (import price, or the missed export revenue when solar
covers it), at least `dhw_min_gap_hours` apart, never inside peak periods or a running legionella cycle.
A charge that already happened today (tank reached target - hysteresis) counts for today.

Legionella (option legionella_mode = plan): the tank must reach `legionella_temp` at least every `legionella_interval_days`.
The deadline is counted from the last time the tank was that hot (restart-proof: from an entity and/or long-term
statistics, see planner.legionella_status). Once the deadline lies inside the horizon (or has passed), one legionella
charge is placed at the lowest incremental cost between
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
from .model import DT, dhw_charge, dhw_cop, dhw_start_temp

SKIP_PENALTY = 25.0   # currency; a day without a possible charge is allowed but strongly avoided


def slot_power(p: dict, To) -> np.ndarray:
    """electric kW during a planned charge (fixed heat per charge, COP depends on the outdoor temperature)"""
    return p[P.DHW_POWER] * p[P.DHW_COP_REF] / dhw_cop(p, To)


def plan_dhw(p: dict, t, To, price, feed, E, PV, peak, busy, tz: ZoneInfo, now: float, done_today: int = 0,
             last_end: float | None = None, running: tuple | None = None, known=None, legionella: dict | None = None,
             tank: float | None = None) -> dict:
    """returns dict(blocks=[(start, end)], kinds={start: kind}, dhw=bool array, dhw_kw=array, dhw_heat=array, details=[...],
    legionella=...)

    running: (start, end[, kind]) of a charge that already started; legionella: dict(earliest=ts, due=ts) when a
    legionella charge has to be planned (legionella_mode plan), else None; tank: current tank temperature (model sizing)"""
    t = np.asarray(t, float)
    N = len(t)
    To = np.asarray(To, float)
    model = p[P.DHW_SIZING] == "model"
    n_fix = max(1, int(round(p[P.DHW_CHARGE_HOURS] / DT)))
    nl_fix = max(1, int(round(p[P.LEGIONELLA_CHARGE_HOURS] / DT)))
    gap = int(round(p[P.DHW_MIN_GAP] / DT))
    per_day = int(p[P.DHW_PER_DAY])
    pw = slot_power(p, To)
    pwl = pw * p[P.LEGIONELLA_POWER] / p[P.DHW_POWER]
    cop_fix = dhw_cop(p, To)
    targets = dict(normal=p[P.DHW_TARGET], hot=p[P.DHW_HOT_TARGET], legionella=p[P.LEGIONELLA_SETPOINT])
    net0 = np.asarray(E, float) - np.asarray(PV, float)
    pr = np.asarray(price, float)
    fd = np.asarray(feed, float)

    def mc(x, a: int = 0):
        b = a + len(x)
        return pr[a:b] * np.maximum(x, 0) - fd[a:b] * np.maximum(-x, 0)

    cache: dict = {}

    def prof(kind: str, k: int):
        """(slots, electric kW per slot, heat kW per slot, estimated start temperature) of a charge of `kind` starting in slot k"""
        key = (kind, k)
        if key not in cache:
            if model:
                T0 = dhw_start_temp(p, tank, (t[k] - now) / 3600.0)
                r = dhw_charge(p, T0, targets[kind], To[k:] if k < N else To[-1:])
                cache[key] = (r["slots"], r["el"], r["heat"], T0)
            else:
                m = nl_fix if kind in ("legionella", "hot") else n_fix
                e = (pwl if kind in ("legionella", "hot") else pw)[k:k + m]
                cache[key] = (m, e, e * cop_fix[k:k + m], None)
        return cache[key]

    def cost_of(kind: str, k: int) -> float:
        m, e, _, _ = prof(kind, k)
        e = e[:max(0, N - k)]
        x = net0[k:k + len(e)]
        return float(DT * (mc(x + e, k) - mc(x, k)).sum())

    blocked = np.asarray(busy, bool).copy()
    if p[P.AVOID_PEAKS]:
        blocked |= np.asarray(peak, bool)
    dhw = np.zeros(N, bool)
    dkw = np.zeros(N)
    dheat = np.zeros(N)
    fixed = []
    kinds: dict[float, str] = {}
    last = -1e9 if last_end is None else (last_end - t[0]) / STEP
    day_of = [dt.datetime.fromtimestamp(x, tz).date() for x in t]
    today = dt.datetime.fromtimestamp(now, tz).date()
    have = {today: min(done_today, per_day)}
    leg_info = None
    planned_leg = None

    def place(k: int, m: int, e, q):
        dhw[k:k + m] = True
        dkw[k:k + m] = e[:max(0, min(m, N - k))]
        dheat[k:k + m] = q[:max(0, min(m, N - k))]

    if running is not None and running[1] > t[0]:
        rk = running[2] if len(running) > 2 and running[2] else "normal"
        k1 = int(min(N, np.ceil((running[1] - t[0]) / STEP)))
        e_run = (pwl if rk in ("legionella", "hot") else pw)[:k1]
        q_run = e_run * cop_fix[:k1]
        if model and tank is not None:
            # remaining part from the current tank temperature (already running: no start-up); extended, never shortened
            r = dhw_charge(p, tank, targets.get(rk, p[P.DHW_TARGET]), To, startup=False)
            kcap = int(np.ceil((running[0] + p[P.DHW_MAX_CHARGE_HOURS] * 3600 - t[0]) / STEP - 1e-9))
            rem = r["slots"] if r["hours"] > 0 else 0
            k1 = int(min(N, max(k1, min(rem, kcap))))
            e_run = np.zeros(k1)
            q_run = np.zeros(k1)
            e_run[:min(k1, len(r["el"]))] = r["el"][:k1]
            q_run[:min(k1, len(r["heat"]))] = r["heat"][:k1]
        place(0, k1, e_run, q_run)
        fixed.append((running[0], t[0] + k1 * STEP))
        kinds[float(running[0])] = rk
        d0 = dt.datetime.fromtimestamp(running[0], tz).date()
        have[d0] = have.get(d0, 0) + 1
        last = max(last, k1)
        blocked[:k1] = True
        if rk == "legionella":
            legionella = None          # the running charge is the legionella charge
            leg_info = dict(start=float(running[0]), end=float(t[0] + k1 * STEP), running=True)

    def free(k: int, m: int) -> bool:
        return k + m <= N and not blocked[k:k + m].any()

    # ------------------------------------------------------------------ legionella charge (before the normal charges)
    if legionella is not None and N > 1:
        due = float(legionella["due"])
        earliest = float(legionella.get("earliest") or t[0])
        horizon_end = t[-1] + STEP
        overdue = due <= now
        lo = max(t[0], earliest)
        hi = (t[0] + 86400) if overdue else due      # latest end of the charge
        if overdue or due <= horizon_end:
            # incremental cost: the legionella charge replaces the cheapest normal charge of its calendar day
            day_best: dict = {}
            for k in range(N):
                if free(k, prof("normal", k)[0]):
                    c = cost_of("normal", k)
                    if c < day_best.get(day_of[k], np.inf):
                        day_best[day_of[k]] = c
            best_k, best_c = None, np.inf
            for k in range(N):
                ml = prof("legionella", k)[0]
                if t[k] < lo - 1e-6 or t[k] + ml * STEP > hi + 1e-6 or not free(k, ml):
                    continue
                c = cost_of("legionella", k) - (day_best.get(day_of[k], 0.0) if have.get(day_of[k], 0) < per_day else 0.0)
                if k < last + gap and not overdue:
                    continue
                if c + 1e-7 * k < best_c:
                    best_k, best_c = k, c + 1e-7 * k
            if best_k is not None:
                k = best_k
                ml, el_, q_, T0 = prof("legionella", k)
                place(k, ml, el_, q_)
                planned_leg = (t[k], t[k] + ml * STEP)
                kinds[float(t[k])] = "legionella"
                have[day_of[k]] = have.get(day_of[k], 0) + 1
                # keep the normal charges of other days at least `gap` away from it (before: by their own length)
                blocked[k:min(N, k + ml + gap)] = True
                for kk in range(max(0, k - gap - int(np.ceil(p[P.DHW_MAX_CHARGE_HOURS] / DT)) - n_fix), k):
                    if kk + prof("normal", kk)[0] + gap > k:
                        blocked[kk] = True
                el = float(el_.sum() * DT)
                leg_info = dict(start=float(t[k]), end=float(t[k] + ml * STEP), kwh=el, cost=cost_of("legionella", k),
                                price=float((pr[k:k + ml] * el_).sum() / max(1e-9, el_.sum())),
                                heat_kwh=float(q_.sum() * DT), start_temp=T0,
                                provisional=bool(known is not None and not np.asarray(known)[k:k + ml].all()))
            else:
                leg_info = dict(start=None, end=None, reason="no free slot before the deadline")
    # candidate starts
    ok = np.zeros(N, bool)
    cost = np.full(N, np.inf)
    for k in range(N):
        if not free(k, prof("normal", k)[0]):
            continue
        ok[k] = True
        cost[k] = cost_of("normal", k)
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
                e = k + prof("normal", k)[0]
                if e not in new or v < new[e][0]:
                    new[e] = (v, path + [k])
        states = new
    best = min(states.values(), key=lambda x: x[0]) if states else (0.0, [])
    blocks = list(fixed) + ([planned_leg] if planned_leg else [])
    details = []
    if leg_info and leg_info.get("start") is not None and not leg_info.get("running"):
        details.append(dict(leg_info, kind="legionella"))
    # hot charges: a cheap normal charge is extended to a hot charge (hot setpoint; longer charge)
    ratio = p[P.DHW_HOT_PRICE_RATIO]
    kn = np.asarray(known, bool) if known is not None else np.ones(N, bool)
    med = float(np.median(pr[kn])) if kn.any() else float(np.median(pr))
    for k in best[1]:
        kind = "normal"
        n = prof("normal", k)[0]
        if ratio > 0:
            mh, eh, _, _ = prof("hot", k)
            if k + mh <= N and not blocked[k + n:k + mh].any() and not dhw[k + n:k + mh].any():
                avg = cost_of("hot", k) / max(1e-9, float(eh.sum() * DT))
                if avg <= ratio * med:
                    kind = "hot"
        m, pk, qk, T0 = prof(kind, k)
        place(k, m, pk, qk)
        blocks.append((t[k], t[k] + m * STEP))
        kinds[float(t[k])] = kind
        el = float(pk.sum() * DT)
        details.append(dict(start=t[k], end=t[k] + m * STEP, kwh=el, cost=cost_of(kind, k),
                            price=float((pr[k:k + m] * pk).sum() / max(1e-9, pk.sum())), heat_kwh=float(qk.sum() * DT), start_temp=T0,
                            provisional=bool(known is not None and not np.asarray(known)[k:k + m].all()), kind=kind))
    details.sort(key=lambda d: d["start"])
    return dict(blocks=sorted(blocks), kinds=kinds, dhw=dhw, dhw_kw=dkw, dhw_heat=dheat, details=details, fixed=fixed, legionella=leg_info)


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
