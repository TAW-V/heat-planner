"""Thermal model of the house and the heat pump (pure numpy).

Two-node RC model with an unmeasured-gains state, discretised exactly for 15-minute slots:

    x = [Tr (room air), Ts (building mass), d (unmeasured gains, kW)]
    Cr dTr/dt = Hrs (Ts - Tr) - Hro (Tr - To) + qi + kl E + g (1 - fs) PV + d
    Cs dTs/dt = Hrs (Tr - Ts) - Hg (Ts - TG) + Q + g fs PV
    dd/dt     = -d / TD

    inputs u = [To, 1, PV kW, Q kW (heat pump space-heating heat), E kW (household electricity), 0]

The room temperature is measured; a steady-state Kalman filter estimates Ts and d from the recent history.
"""
from __future__ import annotations

import math

import numpy as np

from . import params as P

DT = 0.25  # h per slot


def expm(M: np.ndarray) -> np.ndarray:
    """matrix exponential (scaling and squaring + Taylor), good for the small matrices used here"""
    n = max(0, int(math.ceil(math.log2(max(np.abs(M).sum(1).max(), 1e-12)))) + 1)
    X = M / 2 ** n
    E = np.eye(len(M))
    T = np.eye(len(M))
    for k in range(1, 20):
        T = T @ X / k
        E = E + T
    for _ in range(n):
        E = E @ E
    return E


class Model:
    """discrete-time house model for one parameter set (normalised options, see params.normalise)"""

    def __init__(self, p: dict, dt: float = DT):
        self.p = p
        self.dt = dt
        self.Cr = p[P.AIR_CAPACITY]
        self.Cs = p[P.MASS_CAPACITY]
        Hro = p[P.HEAT_LOSS] / 1000.0
        Hrs = p[P.COUPLING] / 1000.0
        Hg = p[P.GROUND_LOSS] / 1000.0
        TG = p[P.GROUND_TEMP]
        qi = p[P.GAIN_FIXED] / 24.0
        kl = p[P.GAIN_FRACTION]
        g = p[P.SOLAR_GAIN]
        fs = p[P.SOLAR_TO_MASS]
        TD = P.DISTURBANCE_HOURS
        Cr, Cs = self.Cr, self.Cs
        self.Hrs = Hrs
        A = np.array([[-(Hro + Hrs) / Cr, Hrs / Cr, 1 / Cr], [Hrs / Cs, -(Hrs + Hg) / Cs, 0], [0, 0, -1 / TD]])
        B = np.array([[Hro / Cr, qi / Cr, g * (1 - fs) / Cr, 0, kl / Cr, 0],
                      [0, Hg * TG / Cs, g * fs / Cs, 1 / Cs, 0, 0], [0] * 6])
        n = 3
        m = B.shape[1]
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A * dt
        M[:n, n:] = B * dt
        E = expm(M)
        self.Ad = E[:n, :n]
        self.Bd = E[:n, n:]
        # steady-state Kalman gain, measurement = room temperature
        Qn = np.diag(P.KALMAN_Q_HOURLY) * dt
        Cm = np.array([[1.0, 0, 0]])
        Pm = np.eye(3)
        K = np.zeros(3)
        for _ in range(500):
            Pp = self.Ad @ Pm @ self.Ad.T + Qn
            S = (Cm @ Pp @ Cm.T)[0, 0] + P.KALMAN_R
            K = (Pp @ Cm.T / S)[:, 0]
            Pm = (np.eye(3) - np.outer(K, Cm[0])) @ Pp
        self.K = K

    @staticmethod
    def U(To, PV, Q, E) -> np.ndarray:
        To = np.asarray(To, float)
        n = len(To)
        return np.stack([To, np.ones(n), np.asarray(PV, float), np.asarray(Q, float), np.asarray(E, float), np.zeros(n)], 1)

    def filter(self, U: np.ndarray, y, x0=None) -> np.ndarray:
        """filtered states x[k|k] (state at the start of slot k after measuring y[k]); NaN = no measurement"""
        y = np.asarray(y, float)
        good = y[~np.isnan(y)]
        if x0 is not None:
            x = np.array(x0, float)
        else:
            t = float(good[0]) if good.size else 20.0
            x = np.array([t, t + 0.5, 0.0])
        X = np.zeros((len(y), 3))
        for k in range(len(y)):
            if k > 0:
                x = self.Ad @ x + self.Bd @ U[k - 1]
            if not np.isnan(y[k]):
                x = x + self.K * (y[k] - x[0])
            X[k] = x
        return X

    def sim(self, x0, U: np.ndarray) -> np.ndarray:
        x = np.array(x0, float)
        X = np.zeros((len(U) + 1, 3))
        X[0] = x
        for k in range(len(U)):
            x = self.Ad @ x + self.Bd @ U[k]
            X[k + 1] = x
        return X


# ------------------------------------------------------------------------------------------------ heat pump
def heat_output(p: dict, To) -> np.ndarray:
    """space-heating heat output at full load (kW) as a function of the outdoor temperature"""
    To = np.asarray(To, float)
    q = p[P.HP_MAX_HEAT] - p[P.HP_HEAT_SLOPE] * np.maximum(To - p[P.HP_KNEE_TEMP], 0)
    return np.clip(q, p[P.HP_MIN_HEAT], max(p[P.HP_MAX_HEAT], p[P.HP_MIN_HEAT]))


def cop(p: dict, To, Q) -> np.ndarray:
    """space-heating COP: base + temp*To - load*Q, clipped to [min, max]"""
    c = p[P.COP_BASE] + p[P.COP_TEMP] * np.asarray(To, float) - p[P.COP_LOAD] * np.asarray(Q, float)
    return np.clip(c, p[P.COP_MIN], p[P.COP_MAX])


def dhw_cop(p: dict, To) -> np.ndarray:
    return np.clip(p[P.DHW_COP_REF] + p[P.DHW_COP_SLOPE] * (np.asarray(To, float) - 7.0), 1.0, 8.0)


def dhw_heat_kw(p: dict, T, To) -> np.ndarray:
    """heat into the tank (kW) during a charge at tank temperature T and outdoor temperature To (model sizing)"""
    q = p[P.DHW_HEAT_KW] - p[P.DHW_HEAT_TANK_SLOPE] * (np.asarray(T, float) - 45.0) + p[P.DHW_HEAT_OUTDOOR_SLOPE] * (np.asarray(To, float) - 7.0)
    return np.maximum(q, P.DHW_MIN_HEAT_KW)


def dhw_charge_cop(p: dict, T, To) -> np.ndarray:
    """COP of a tank charge at tank temperature T and outdoor temperature To (model sizing)"""
    c = (p[P.DHW_CHARGE_COP] - p[P.DHW_CHARGE_COP_TANK_SLOPE] * (np.asarray(T, float) - 45.0)
         + p[P.DHW_CHARGE_COP_OUTDOOR_SLOPE] * (np.asarray(To, float) - 7.0))
    return np.clip(c, *P.DHW_COP_LIMITS)


def dhw_charge(p: dict, T0: float, target: float, To, startup: bool = True, sub: int = 5) -> dict:
    """simulate one tank charge from tank temperature T0 to `target` in 15-minute slots (slot 0 = start of the charge).

    To: outdoor temperature per slot (the last value is repeated). Returns dict(hours = time to reach the target without
    margin, slots = planned length (hours x (1 + margin), rounded up to whole slots, 1..max), el = electric kW per planned slot,
    heat = heat kW per planned slot, heat_kwh, el_kwh, reached = target reached within the maximum length)."""
    To = np.atleast_1d(np.asarray(To, float))
    cap = p[P.DHW_TANK_CAPACITY]
    kmax = max(1, int(math.ceil(p[P.DHW_MAX_CHARGE_HOURS] / DT - 1e-9)))
    h = DT / sub
    wait = (p[P.DHW_START_MINUTES] / 60.0) if startup else 0.0
    T = float(T0)
    el = np.zeros(kmax)      # kWh per slot (converted to kW below)
    heat = np.zeros(kmax)
    need = 0.0 if T >= target else None
    q0, qs, qo = p[P.DHW_HEAT_KW], p[P.DHW_HEAT_TANK_SLOPE], p[P.DHW_HEAT_OUTDOOR_SLOPE]
    c0, cs, co = p[P.DHW_CHARGE_COP], p[P.DHW_CHARGE_COP_TANK_SLOPE], p[P.DHW_CHARGE_COP_OUTDOOR_SLOPE]
    clo, chi = P.DHW_COP_LIMITS
    k = 0
    while need is None and k < kmax:
        to = float(To[min(k, len(To) - 1)])
        for j in range(sub):
            tm = (k * sub + j) * h
            q = max(q0 - qs * (T - 45.0) + qo * (to - 7.0), P.DHW_MIN_HEAT_KW)       # = dhw_heat_kw (scalar)
            pel = q / min(max(c0 - cs * (T - 45.0) + co * (to - 7.0), clo), chi)   # = q / dhw_charge_cop
            ws = min(h, max(0.0, wait - tm))          # start-up part of this step: power, no heat into the tank yet
            el[k] += P.DHW_START_POWER * pel * ws
            act = h - ws
            if act <= 0:
                continue
            dT = q * act / cap
            part = 1.0 if T + dT < target else (target - T) / dT
            T = T + dT * part
            heat[k] += q * act * part
            el[k] += pel * act * part
            if part < 1.0 or T >= target:
                need = tm + ws + act * part
                break
        k += 1
    el /= DT
    heat /= DT
    reached = need is not None
    hours = need if reached else kmax * DT
    n = int(math.ceil(hours * (1 + p[P.DHW_CHARGE_MARGIN] / 100.0) / DT - 1e-6))
    n = min(kmax, max(1, n))
    return dict(hours=hours, slots=n, el=el[:n], heat=heat[:n], heat_kwh=float(heat[:n].sum() * DT), el_kwh=float(el[:n].sum() * DT),
                reached=reached, start_temp=float(T0), target=float(target))


def dhw_start_temp(p: dict, tank: float | None, hours_ahead: float) -> float:
    """expected tank temperature at the start of a charge `hours_ahead` from now: the current temperature cooled by standing
    losses for a charge that starts within DHW_KNOWN_HOURS, otherwise at most the typical start temperature (hot-water use in
    between is unknown) but not below dhw_low_temp - dhw_hysteresis; without a tank temperature the typical start temperature"""
    typ = p[P.DHW_START_TEMP]
    if tank is None:
        return typ
    h = max(0.0, hours_ahead)
    cooled = P.DHW_ROOM_TEMP + (tank - P.DHW_ROOM_TEMP) * math.exp(-h / P.DHW_LOSS_TAU_HOURS)
    if h <= P.DHW_KNOWN_HOURS:
        return cooled
    # the heat pump keeps the tank above low - hysteresis by itself
    return max(min(cooled, typ), min(typ, p[P.DHW_LOW] - p[P.DHW_HYST]))


def margin(hours, m24: float) -> np.ndarray:
    """horizon-dependent safety margin on the comfort floor: 0 now, 0.4*m24 at 6 h, 0.6*m24 at 12 h, m24 from 24 h"""
    return np.interp(hours, [0, 6, 12, 24, 1e9], [0.0, 0.4 * m24, 0.6 * m24, m24, m24])
