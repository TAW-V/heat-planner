"""Offline tests of the planning engine with synthetic data (no Home Assistant needed).

Run with `python -m pytest tests` or `python tests/test_engine.py`.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sys
import types
from zoneinfo import ZoneInfo

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "custom_components", "heat_planner")
# load the engine package without importing the Home Assistant parts of the integration
pkg = types.ModuleType("heat_planner")
pkg.__path__ = [ROOT]
sys.modules.setdefault("heat_planner", pkg)

from heat_planner.engine import params as P  # noqa: E402
from heat_planner.engine import planner  # noqa: E402
from heat_planner.engine.inputs import forecast_adder, forecast_price_dict, price_dict, prices_from_attributes  # noqa: E402

TZ = "Europe/Amsterdam"
OPTIONS = {
    P.ROOM_TEMP_ENTITY: "sensor.room_temperature",
    P.OUTDOOR_TEMP_ENTITY: "sensor.outdoor_temperature",
    P.WEATHER_ENTITY: "weather.home",
    P.HEAT_OUTPUT_ENTITY: "sensor.heat_pump_heat_power",
    P.PV_POWER_ENTITY: "sensor.pv_power",
    P.PV_FORECAST_ENTITIES: ["sensor.pv_forecast_today", "sensor.pv_forecast_tomorrow"],
    P.IMPORT_ENERGY_ENTITY: "sensor.grid_import",
    P.EXPORT_ENERGY_ENTITY: "sensor.grid_export",
    P.PV_ENERGY_ENTITY: "sensor.pv_energy",
    P.HP_ENERGY_ENTITY: "sensor.heat_pump_energy",
    P.PRICE_ENTITY: "sensor.electricity_price",
    P.BATTERY_SOC_ENTITY: "sensor.battery_energy",
    P.DHW_TANK_ENTITY: "sensor.tank_temperature",
}


def synthetic_raw(now: float | None = None) -> dict:
    """two days of history and ~40 h of quarter-hour prices for a mild autumn day"""
    tz = ZoneInfo(TZ)
    if now is None:
        now = dt.datetime(2026, 11, 3, 14, 7, tzinfo=tz).timestamp()
    t5 = np.arange(now - 2 * 86400, now, 300)
    hod = np.array([dt.datetime.fromtimestamp(x, tz).hour + dt.datetime.fromtimestamp(x, tz).minute / 60 for x in t5])
    out_t = 8 + 3 * np.sin((hod - 9) / 24 * 2 * math.pi)
    room = 20 + 0.4 * np.sin((hod - 15) / 24 * 2 * math.pi)
    pv = np.maximum(0, 4000 * np.sin((hod - 8) / 9 * math.pi)) * ((hod > 8) & (hod < 17))
    heat = np.where((hod > 11) & (hod < 14), 8.0, 0.0)
    ms = lambda a: [dict(start=float(x) * 1000, mean=float(v), max=float(v)) for x, v in zip(t5, a)]
    th = np.arange((now - 8 * 86400) // 3600 * 3600, now, 3600)
    hh = np.array([dt.datetime.fromtimestamp(x, tz).hour for x in th])
    hourly = lambda a: [dict(start=float(x) * 1000, change=float(v), mean=float(v)) for x, v in zip(th, a)]
    price_h = 0.25 + 0.10 * np.isin(hh, [7, 8, 17, 18, 19]) - 0.06 * np.isin(hh, [1, 2, 3, 4, 12, 13])
    d0 = dt.datetime.fromtimestamp(now, tz).replace(hour=0, minute=0, second=0, microsecond=0)
    tq = np.arange(d0.timestamp(), d0.timestamp() + 2 * 86400, 900)
    hq = np.array([dt.datetime.fromtimestamp(x, tz).hour for x in tq])
    pq = 0.25 + 0.10 * np.isin(hq, [7, 8, 17, 18, 19]) - 0.06 * np.isin(hq, [1, 2, 3, 4, 12, 13])
    fc = [dict(datetime=dt.datetime.fromtimestamp(x, tz).isoformat(), temperature=8 + 3 * math.sin((dt.datetime.fromtimestamp(x, tz).hour - 9) / 24 * 2 * math.pi))
          for x in np.arange(now // 3600 * 3600, now + 72 * 3600, 3600)]
    det = [dict(period_start=dt.datetime.fromtimestamp(x, tz).isoformat(),
                pv_estimate=max(0.0, 3.0 * math.sin((dt.datetime.fromtimestamp(x, tz).hour - 8) / 9 * math.pi)) if 8 < dt.datetime.fromtimestamp(x, tz).hour < 17 else 0.0)
           for x in np.arange(d0.timestamp(), d0.timestamp() + 2 * 86400, 1800)]
    st = lambda s, unit=None, **a: dict(s=str(s), a=dict(a, **({"unit_of_measurement": unit} if unit else {})), lc=now - 3600)
    states = {
        "sensor.room_temperature": st(20.1, "°C"), "sensor.outdoor_temperature": st(9.0, "°C"), "weather.home": st("cloudy", temperature_unit="°C"),
        "sensor.heat_pump_heat_power": st(0, "kW"), "sensor.pv_power": st(1500, "W"),
        "sensor.pv_forecast_today": st(10, "kWh", detailedForecast=det[:48]), "sensor.pv_forecast_tomorrow": st(10, "kWh", detailedForecast=det[48:]),
        "sensor.grid_import": st(1000, "kWh"), "sensor.grid_export": st(500, "kWh"), "sensor.pv_energy": st(900, "kWh"),
        "sensor.heat_pump_energy": st(300, "kWh"), "sensor.electricity_price": st(0.3, "EUR/kWh"), "sensor.battery_energy": st(3.0, "kWh"),
        "sensor.tank_temperature": st(42, "°C"),
    }
    return dict(now=now, tz=TZ, states=states, prices=[dict(start=float(x), price=float(v)) for x, v in zip(tq, pq)], weather=fc,
                stats_hour={"sensor.grid_import": hourly(np.full(len(th), 0.6)), "sensor.grid_export": hourly(np.zeros(len(th))),
                            "sensor.pv_energy": hourly(np.zeros(len(th))), "sensor.heat_pump_energy": hourly(np.full(len(th), 0.1)),
                            "sensor.electricity_price": hourly(price_h)},
                stats_5min={"sensor.room_temperature": ms(room), "sensor.outdoor_temperature": ms(out_t),
                            "sensor.heat_pump_heat_power": ms(heat), "sensor.pv_power": ms(pv), "sensor.tank_temperature": ms(np.full(len(t5), 42.0))})


def test_heating_plan_keeps_comfort():
    r = planner.run(synthetic_raw(), dict(OPTIONS))
    assert r["tr_min"] >= P.NUMBERS[P.HARD_MIN_TEMP][0]
    assert r["tr_max"] <= P.NUMBERS[P.MAX_TEMP][0] + 0.3
    assert "battery" not in r
    assert r["hot_water"]["mode"] == "off"
    assert isinstance(r["schedule"], str)


def test_joint_battery_and_hot_water():
    o = dict(OPTIONS, **{P.PLAN_BATTERY: True, P.DHW_MODE: "plan"})
    r = planner.run(synthetic_raw(), o, lang="en")
    b = r["battery"]
    assert len(b["soc"]) == len(r["tr"])
    cap = P.NUMBERS[P.BAT_CAPACITY][0]
    assert all(-1e-9 <= s <= cap + 1e-9 for s in b["soc"])
    hw = r["hot_water"]
    assert hw["mode"] == "plan" and hw["blocks"], hw
    # the planned charges avoid the most expensive quarters
    for blk in hw["blocks"]:
        assert dt.datetime.fromisoformat(blk["start"]).hour not in (7, 8, 17, 18, 19)


def test_defaults_are_stable():
    p = P.normalise({})
    assert p[P.MIN_TEMP] == 18.5 and p[P.MAX_TEMP] == 22.0 and len(p["_bias"]) == 24


def test_price_parsers():
    hourly = [dict(start=f"2026-11-03T{h:02d}:00:00+01:00", price=0.1 * h) for h in range(3)]
    d = price_dict(hourly)
    assert len(d) == 12   # hourly prices are spread over the quarters
    attrs = {"raw_today": [dict(start="2026-11-03T00:00:00+01:00", end="2026-11-03T01:00:00+01:00", value=25.0)]}
    assert prices_from_attributes(attrs, "c/kWh")[0]["price"] == 0.25


def test_price_forecast():
    attrs = {"raw_today": [dict(start=f"2026-11-03T{h:02d}:00:00+01:00", value=100.0) for h in range(3)],
             "raw_forecast": [dict(start="2026-11-03T02:00:00+01:00", value=999.0),
                              dict(start="2026-11-03T03:00:00+01:00", value=200.0)]}
    spot = forecast_price_dict(attrs, "EUR/MWh")
    assert len(spot) == 16 and max(spot.values()) == 0.2   # published values win over the forecast
    known = {k: v * 1.21 + 0.13 for k, v in spot.items() if v == 0.1}
    adder, n = forecast_adder(known, spot, 0.21, 0.0)
    assert n == 12 and abs(adder - 0.13) < 1e-9
    assert forecast_adder({}, spot, 0.21, 0.05) == (0.05, 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
