"""Dynamic-programming optimiser for space heating (+ optional home battery) over 15-minute slots.

State: room temperature x (mass - room) grid (bilinear interpolation of the value function), block run-length
(minimum block length) and, in joint mode, the battery state of charge on a 0.2 kWh grid.
Decisions per slot: heating off / start / run (full-load heat output, no modulation) and battery charge/discharge.
Objective: import cost - export revenue + battery wear + start cost + comfort penalty - terminal value of stored
heat and battery energy. The forward pass re-simulates the exact 3-state model with the value function; the
backward pass is checkpointed to keep memory small.
"""
from __future__ import annotations

import math

import numpy as np

from . import params as P
from .model import DT, Model, cop, heat_output, margin

# comfort penalty weights (currency per K per slot) and other solver constants
# comfort penalties come from the options margin_penalty (inside the safety margin) and comfort_penalty
# (below the floor or above the ceiling, x10 below the hard minimum)
TERMINAL_FRACTION = 0.5  # stored heat/energy at the end is valued at this share of the expected cost of producing it
TR_STEP = 0.1
DL_STEP = 0.2
SOC_STEP = 0.2
CHECKPOINT = 12


def battery_options(p: dict) -> dict:
    cap = p[P.BAT_CAPACITY]
    return dict(cap=cap, soc_min=cap * p[P.BAT_MIN_SOC] / 100, soc_max=cap * p[P.BAT_MAX_SOC] / 100,
                p_ch=p[P.BAT_CHARGE_KW], p_dis=p[P.BAT_DISCHARGE_KW], eta_c=p[P.BAT_CHARGE_EFF],
                eta_d=p[P.BAT_DISCHARGE_EFF], wear=p[P.BAT_WEAR], export=p[P.BAT_EXPORT], ds=SOC_STEP)


def plan(p: dict, model: Model, inp: dict, x0, r0: int = 0, soc0: float = 0.0, battery: dict | None = None,
         force=None) -> dict:
    """inp: equal-length arrays per 15-min slot starting at the time of x0:
         To, PV, E (kW), price, feed (currency/kWh), peak (bool: no heating), dhw (bool: heat pump busy with hot water),
         dhw_kw (electric kW of hot water), hours (hours ahead, for the margin)
       x0 = [Tr, Ts, d]; r0 = slots the current block already ran; soc0 = battery kWh; battery = battery_options(p) or None
       force = optional bool array: slots in which heating must run (the rest of a block that already started)"""
    floor0 = p[P.MIN_TEMP]
    ceil = p[P.MAX_TEMP]
    hard = p[P.HARD_MIN_TEMP]
    pen_soft = float(p[P.MARGIN_PENALTY])
    pen_hard = float(p[P.COMFORT_PENALTY])
    tr_lo = hard - 0.2
    tr_hi = ceil + 0.6
    N = len(inp["To"])
    fon = np.zeros(N, bool) if force is None else np.asarray(force, bool)[:N]
    if fon.size < N:
        fon = np.concatenate([fon, np.zeros(N - fon.size, bool)])
    L = max(1, int(round(p[P.MIN_BLOCK_HOURS] / DT)))
    R = L + 1
    ramp = p[P.START_RAMP]
    To = np.asarray(inp["To"], float)
    PV = np.asarray(inp["PV"], float)
    E = np.asarray(inp["E"], float)
    pr = np.asarray(inp["price"], float)
    f = np.asarray(inp["feed"], float)
    peak = np.asarray(inp["peak"], bool)
    dhw = np.asarray(inp["dhw"], bool)
    dkw = np.asarray(inp["dhw_kw"], float)
    base_floor = np.full(N, floor0)
    floor = base_floor + margin(np.asarray(inp["hours"], float) + DT, p[P.SAFETY_MARGIN])
    Qf = heat_output(p, To)
    Q = np.where(dhw, 0.0, Qf)
    Pel_run = np.where(dhw, 0.0, Q / cop(p, To, Qf))
    Pel_start = Pel_run * ramp
    Ad, Bd = model.Ad, model.Bd
    ad = Ad[2, 2]
    dk = x0[2] * ad ** np.arange(N)
    U0 = Model.U(To, PV, np.zeros(N), E)
    b = (U0 @ Bd[:2].T) + np.outer(dk, Ad[:2, 2])
    gq = Bd[:2, 3]
    A2 = Ad[:2, :2]
    # grids
    dl_hi = max(4.8, math.ceil((float(np.max(Qf, initial=0.0)) / max(model.Hrs, 1e-6) + 0.6) / DL_STEP) * DL_STEP)
    trg = np.arange(tr_lo, tr_hi + 1e-9, TR_STEP)
    dlg = np.arange(-0.6, dl_hi + 1e-9, DL_STEP)
    nT, nD = len(trg), len(dlg)
    NT = nT * nD
    GT, GD = np.meshgrid(trg, dlg, indexing="ij")
    GT = GT.ravel()
    GS = GT + GD.ravel()
    bat = battery
    if bat is not None:
        ds = bat["ds"]
        S = int(round(bat["cap"] / ds)) + 1
        socg = np.arange(S) * ds
        Jc = int(math.floor(bat["p_ch"] * DT * bat["eta_c"] / ds + 1e-9))
        Jd = int(math.floor(bat["p_dis"] * DT / bat["eta_d"] / ds + 1e-9))
        js = np.arange(-Jd, Jc + 1)
        sok = (socg >= bat["soc_min"] - 1e-9) & (socg <= bat["soc_max"] + 1e-9)
    else:
        ds = 1.0
        S = 1
        js = np.array([0])
        socg = np.zeros(1)
        sok = np.ones(1, bool)
    INF = 1e9
    Cr, Cs = model.Cr, model.Cs
    tail = slice(max(0, N - 96), N)
    vh = TERMINAL_FRACTION * float(np.median(pr[tail] / cop(p, To[tail], Qf[tail])))
    VN = (-vh * (Cr * (GT - floor[-1]) + Cs * (GS - floor[-1])))[None, None, :] \
        + pen_soft * 4 * np.maximum(0, floor[-1] - GT)[None, None, :]
    VN = np.broadcast_to(VN, (R, S, NT)).copy()
    vb = 0.0
    if bat:
        vb = max(0.0, TERMINAL_FRACTION * float(np.median(pr[tail])) * bat["eta_d"] - bat["wear"])
        VN -= vb * socg[None, :, None]

    def interp_idx(Trn, Tsn):
        x = (np.clip(Trn, trg[0], trg[-1]) - trg[0]) / TR_STEP
        y = (np.clip(Tsn - Trn, dlg[0], dlg[-1]) - dlg[0]) / DL_STEP
        i = np.minimum(x.astype(int), nT - 2)
        j = np.minimum(y.astype(int), nD - 2)
        fx = x - i
        fy = y - j
        base = i * nD + j
        return (base, base + 1, base + nD, base + nD + 1), ((1 - fx) * (1 - fy), (1 - fx) * fy, fx * (1 - fy), fx * fy)

    def pen(k, Trn):
        return (pen_soft * np.maximum(0, floor[k] - Trn)
                + pen_hard * (np.maximum(0, base_floor[k] - Trn) + np.maximum(0, Trn - ceil) + 10 * np.maximum(0, hard - Trn)
                                  + 5 * (np.maximum(0, Trn - tr_hi) + np.maximum(0, tr_lo - Trn))))

    if bat:
        bk_all = np.where(js > 0, js * ds / bat["eta_c"], js * ds * bat["eta_d"]) / DT   # grid-side kW (+ = charge)
    else:
        bk_all = np.zeros(1)

    def slot_cost(k, pel):
        base = E[k] + dkw[k] + pel - PV[k]
        net = base + bk_all
        c = DT * (pr[k] * np.maximum(net, 0) - f[k] * np.maximum(-net, 0))
        if bat:
            c = c + bat["wear"] * np.where(js < 0, -js * ds * bat["eta_d"], 0)
            if not bat["export"]:
                c = np.where((js < 0) & (net < min(0.0, base) - 1e-9), INF, c)
        return c

    acts = ("off", "start", "run")

    def backward_step(k, Vn):
        res = {}
        for a in acts:
            q = 0.0 if a == "off" else (Q[k] * (ramp if a == "start" else 1.0))
            Trn = A2[0, 0] * GT + A2[0, 1] * GS + b[k, 0] + gq[0] * q
            Tsn = A2[1, 0] * GT + A2[1, 1] * GS + b[k, 1] + gq[1] * q
            idx, w = interp_idx(Trn, Tsn)
            res[a] = (idx, w, pen(k, Trn))

        def W(a, rn):
            idx, w, _ = res[a]
            V = Vn[rn]
            return w[0] * V[..., idx[0]] + w[1] * V[..., idx[1]] + w[2] * V[..., idx[2]] + w[3] * V[..., idx[3]]

        def batmin(Wa, a):
            pel = 0.0 if a == "off" else (Pel_start[k] if a == "start" else Pel_run[k])
            c = slot_cost(k, pel)
            if S == 1:
                return Wa + c[0]
            out = np.full(Wa.shape, INF)
            for jj, j in enumerate(js):
                if c[jj] >= INF:
                    continue
                if j >= 0:
                    src = Wa[..., j:, :]
                    dst = (Ellipsis, slice(0, S - j), slice(None))
                else:
                    src = Wa[..., :S + j, :]
                    dst = (Ellipsis, slice(-j, S), slice(None))
                np.minimum(out[dst], src + c[jj], out=out[dst])
            out[..., ~sok, :] = INF
            return out

        V = np.full((R, S, NT), INF)
        if fon[k]:   # forced: the block keeps running (or starts) in this slot, also during a peak
            V[0] = batmin(W("start", 1 if L > 1 else L), "start") + res["start"][2] + p[P.START_COST]
            if L > 1:
                rr = np.minimum(np.arange(1, L) + 1, L)
                V[1:L] = batmin(W("run", rr), "run") + res["run"][2]
            V[L] = batmin(W("run", L), "run") + res["run"][2]
            return V
        Moff0 = batmin(W("off", 0), "off") + res["off"][2]
        V[0] = Moff0
        if not peak[k]:
            rn = 1 if L > 1 else L
            V[0] = np.minimum(V[0], batmin(W("start", rn), "start") + res["start"][2] + p[P.START_COST])
            if L > 1:
                rr = np.minimum(np.arange(1, L) + 1, L)
                V[1:L] = batmin(W("run", rr), "run") + res["run"][2]
            V[L] = np.minimum(Moff0, batmin(W("run", L), "run") + res["run"][2])
        else:
            V[L] = Moff0   # a block may only be stopped by a peak once it reached the minimum length
        return V

    C = CHECKPOINT
    ck = {N: VN}
    V = VN
    for k in range(N - 1, -1, -1):
        V = backward_step(k, V)
        if k % C == 0:
            ck[k] = V
    # forward pass
    x = np.array(x0, float)
    r = int(min(r0, L))
    si = int(np.clip(int(round(soc0 / ds)), 0, S - 1)) if bat else 0
    out = dict(u=np.zeros(N, int), Qd=np.zeros(N), Pel=np.zeros(N), Tr=np.zeros(N + 1), Ts=np.zeros(N + 1), soc=np.zeros(N + 1),
               bat=np.zeros(N), net=np.zeros(N), cost=np.zeros(N))
    out["Tr"][0], out["Ts"][0] = x[0], x[1]
    out["soc"][0] = socg[si] if bat else 0
    for c0 in range(0, N, C):
        c1 = min(c0 + C, N)
        seg = {c1: ck[c1]}
        V = ck[c1]
        for k in range(c1 - 1, c0, -1):
            V = backward_step(k, V)
            seg[k] = V
        for k in range(c0, c1):
            Vn = seg[k + 1]
            best = (INF, None)
            if fon[k]:
                cand = [("start", 1 if L > 1 else L)] if r == 0 else [("run", min(r + 1, L))]
            elif r == 0:
                cand = [("off", 0)] + ([] if peak[k] else [("start", 1 if L > 1 else L)])
            elif r < L:
                cand = [] if peak[k] else [("run", min(r + 1, L))]
            else:
                cand = [("off", 0)] + ([] if peak[k] else [("run", L)])
            if not cand:
                cand = [("off", 0)]
            for a, rn in cand:
                q = 0.0 if a == "off" else Q[k] * (ramp if a == "start" else 1.0)
                xn = Ad @ x + Bd @ np.array([To[k], 1, PV[k], q, E[k], 0])
                idx, w = interp_idx(np.array([xn[0]]), np.array([xn[1]]))
                pel = 0.0 if a == "off" else (Pel_start[k] if a == "start" else Pel_run[k])
                c = slot_cost(k, pel)
                pn = pen(k, np.array([xn[0]]))[0] + (p[P.START_COST] if a == "start" else 0)
                for jj, j in enumerate(js):
                    sn = si + j
                    if not (0 <= sn < S) or not sok[sn] or c[jj] >= INF:
                        continue
                    v = sum(w[i][0] * Vn[rn, sn, idx[i][0]] for i in range(4)) + c[jj] + pn
                    if v < best[0]:
                        best = (v, (a, rn, j, xn, q, pel, c[jj]))
            if best[1] is None:   # cannot happen with a feasible 'stay' action, kept as a guard
                best = (INF, ("off", 0, 0, Ad @ x + Bd @ np.array([To[k], 1, PV[k], 0, E[k], 0]), 0.0, 0.0, 0.0))
            a, rn, j, xn, q, pel, cc = best[1]
            out["u"][k] = 0 if a == "off" else 1
            out["Qd"][k] = q
            out["Pel"][k] = pel
            if bat:
                out["bat"][k] = (j * ds / bat["eta_c"] if j > 0 else j * ds * bat["eta_d"]) / DT
            out["net"][k] = E[k] + dkw[k] + pel - PV[k] + out["bat"][k]
            out["cost"][k] = cc
            x = xn
            r = rn
            si = si + j
            out["Tr"][k + 1], out["Ts"][k + 1] = x[0], x[1]
            out["soc"][k + 1] = socg[si] if bat else 0
    out.update(floor=floor, base_floor=base_floor, vh=vh, vb=vb)
    return out


def blocks(u, t0: float, step: int = 900) -> list[tuple[float, float]]:
    """(start_ts, end_ts) of consecutive u == 1 runs"""
    out = []
    k = 0
    n = len(u)
    while k < n:
        if u[k]:
            j = k
            while j < n and u[j]:
                j += 1
            out.append((t0 + k * step, t0 + j * step))
            k = j
        else:
            k += 1
    return out


def elec_cost(net, price, feed) -> float:
    net = np.asarray(net, float)
    return float(np.sum(DT * (price * np.maximum(net, 0) - feed * np.maximum(-net, 0))))
