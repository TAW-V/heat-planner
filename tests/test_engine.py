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



def test_comfort_penalties_configurable():
    p = P.normalise({})
    assert p[P.MARGIN_PENALTY] == 2.0 and p[P.COMFORT_PENALTY] == 20.0
    base = planner.run(synthetic_raw(), dict(OPTIONS))
    same = planner.run(synthetic_raw(), dict(OPTIONS, **{P.MARGIN_PENALTY: 2.0, P.COMFORT_PENALTY: 20.0}))
    assert same["schedule"] == base["schedule"]
    firm = planner.run(synthetic_raw(), dict(OPTIONS, **{P.MARGIN_PENALTY: 50.0}))
    assert firm["heat_kwh"] >= base["heat_kwh"] - 1e-6   # a firmer margin never plans less heat


def test_running_block_is_kept():
    raw = synthetic_raw()
    now = raw["now"]
    t0 = now // 900 * 900
    start, end = t0 - 2 * 900, t0 + 7 * 900          # started 30 min ago, planned until 1 h 45 min from now
    res = planner.run(raw, dict(OPTIONS), previous={"_heat_blocks": [[start, end]]})
    first = res["blocks"][0]
    s = dt.datetime.fromisoformat(first[0]).timestamp()
    assert s == start                                   # reported from its original start
    assert s + first[1] * 3600 >= end - 1e-6            # and kept at least until its planned end
    assert res["running_block"] is not None
    assert res["next_block_start"] == first[0]
    # without a previous plan nothing is forced
    free = planner.run(raw, dict(OPTIONS))
    assert free["running_block"] is None
    # a block that already ended is not forced again
    old = planner.run(raw, dict(OPTIONS), previous={"_heat_blocks": [[t0 - 12 * 900, t0 - 900]]})
    assert old["running_block"] is None and old["schedule"] == free["schedule"]


def test_day_accounting():
    from heat_planner.engine import accounting as A
    raw = synthetic_raw()
    tz = ZoneInfo(TZ)
    now = raw["now"]
    d0 = dt.datetime.fromtimestamp(now, tz).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    t5 = np.arange(d0 - 300, now // 900 * 900, 300)
    hod = np.array([dt.datetime.fromtimestamp(x, tz).hour for x in t5])
    imp = np.where(hod < 9, 0.05, 0.0)
    exp_ = np.where((hod >= 10) & (hod < 14), 0.15, 0.0)
    pv = np.where((hod >= 9) & (hod < 15), 0.25, 0.0)
    hp = np.where((hod >= 2) & (hod < 3), 0.15, 0.0)        # hot water at night (space heating off)
    rows = lambda a, key="change": [dict(start=float(x) * 1000, **{key: float(v)}) for x, v in zip(t5, a)]
    raw["stats_today"] = {"sensor.grid_import": rows(imp), "sensor.grid_export": rows(exp_), "sensor.pv_energy": rows(pv),
                          "sensor.heat_pump_energy": rows(hp), "sensor.battery_energy": rows(np.full(len(t5), 3.0), "mean")}
    opts = dict(OPTIONS, heating_active_entity="sensor.heating_active")
    raw["stats_5min"]["sensor.heating_active"] = [dict(start=float(x) * 1000, mean=0.0, max=0.0) for x in t5]
    r = planner.run(raw, opts)
    td = r["today"]
    rl, rest, tot = td["realised"], td["rest"], td["total"]
    sel = t5 >= d0
    assert abs(rl["import_kwh"] - imp[sel].sum()) < 0.06 and abs(rl["export_kwh"] - exp_[sel].sum()) < 0.06
    assert abs(rl["boiler_kwh"] - hp[sel].sum()) < 0.06 and rl["heating_kwh"] < 0.06
    for part in (rl, rest, tot):
        s = sum(part[k] for k in ("house", "boiler", "heating", "solar_self", "solar_export", "battery"))
        assert abs(s - part["total"]) < 0.06 and abs(part["total"] - part["net"]) < 0.02
        assert abs(part["import_cost"] - part["export_revenue"] - part["net"]) < 0.02
    assert abs(tot["net"] - rl["net"] - rest["net"]) < 0.02
    assert td["realised_until"] == td["rest_from"]
    assert abs(rest["net"] - r["cost"]) < 0.02 or td["tomorrow"] is not None
    # without statistics for today the plan part still works
    raw.pop("stats_today")
    assert planner.run(raw, opts)["today"]["realised"] is None


def test_post_check_never_worse():
    """the exact post-check only replaces the DP schedule when it is cheaper, and keeps the outputs consistent"""
    for o in (dict(OPTIONS), dict(OPTIONS, **{P.PLAN_BATTERY: True, P.DHW_MODE: "plan"})):
        r = planner.run(synthetic_raw(), o)
        pc = r["post_check"]
        assert "error" not in pc, pc
        assert pc["final_cost"] <= pc["dp_cost"] + 1e-9
        assert pc["runtime_ms"] < 2000
        assert len(r["tr"]) == len(r["ts"])
        for _, hours in r["blocks"] + r["blocks_tail"]:
            assert hours > 0


def test_live_battery_limits():
    st = {"number.power": dict(s="0", a=dict(unit_of_measurement="W", max_charge_power=2400, max_discharge_power="N/A")),
          "sensor.capacity": dict(s="7.1", a=dict(unit_of_measurement="kWh"))}
    p = P.normalise({P.BAT_CAPACITY_ENTITY: "sensor.capacity", P.BAT_CHARGE_POWER_ENTITY: "number.power",
                     P.BAT_CHARGE_POWER_ATTR: "max_charge_power", P.BAT_DISCHARGE_POWER_ENTITY: "number.power",
                     P.BAT_DISCHARGE_POWER_ATTR: "max_discharge_power", P.BAT_DISCHARGE_KW: 2.0})
    used = planner.live_battery(p, st)
    assert abs(p[P.BAT_CAPACITY] - 7.1) < 1e-9 and abs(p[P.BAT_CHARGE_KW] - 2.4) < 1e-9
    assert p[P.BAT_DISCHARGE_KW] == 2.0 and P.BAT_DISCHARGE_KW not in used   # unavailable -> configured number
    p = P.normalise({})
    assert planner.live_battery(p, st) == {} and p[P.BAT_CHARGE_KW] == P.NUMBERS[P.BAT_CHARGE_KW][0]


def test_live_battery_soc_limits():
    st = {"number.discharge_limit": dict(s="3", a=dict(unit_of_measurement="%")),
          "number.backup_reserve": dict(s="unknown", a=dict(unit_of_measurement="%")),
          "sensor.reserve_kwh": dict(s="1.4", a=dict(unit_of_measurement="kWh")),
          "number.charge_limit": dict(s="90", a=dict(unit_of_measurement="%"))}
    o = {P.BAT_CAPACITY: 7.0, P.BAT_MIN_SOC: 10.0, P.BAT_MAX_SOC: 95.0,
         P.BAT_MIN_SOC_ENTITIES: ["number.discharge_limit", "number.backup_reserve"], P.BAT_MAX_SOC_ENTITY: "number.charge_limit"}
    p = P.normalise(o)
    used = planner.live_battery(p, st)
    assert p[P.BAT_MIN_SOC] == 3.0 and p[P.BAT_MAX_SOC] == 90.0          # unknown backup reserve ignored
    assert used[P.BAT_MIN_SOC] == "number.discharge_limit"
    p = P.normalise(dict(o, **{P.BAT_MIN_SOC_ENTITIES: ["number.discharge_limit", "sensor.reserve_kwh"]}))
    planner.live_battery(p, st)
    assert abs(p[P.BAT_MIN_SOC] - 20.0) < 1e-9                         # 1.4 kWh of 7 kWh, the highest value wins
    p = P.normalise(dict(o, **{P.BAT_MIN_SOC_ENTITIES: ["number.backup_reserve"], P.BAT_MAX_SOC_ENTITY: "sensor.missing"}))
    assert planner.live_battery(p, st) == {} and p[P.BAT_MIN_SOC] == 10.0 and p[P.BAT_MAX_SOC] == 95.0


def test_house_changes_battery_and_glitches():
    from heat_planner.engine import profile as HP
    rows = lambda vals: [dict(start=float(i * 3600) * 1000, change=v) for i, v in enumerate(vals)]
    stats = {"imp": rows([1.0, 1.0, 1.0, 900.0]), "exp": rows([0.0, 0.5, 0.0, 0.0]), "hp": rows([0.2, 0.2, None, 0.2]),
             "bc": [dict(start=3600 * 1000.0, change=0.3)], "bd": [dict(start=0.0, change=0.4)]}
    ids = {"import": "imp", "export": "exp", "heat_pump": "hp", "battery_charge": "bc", "battery_discharge": "bd"}
    h = HP.house_changes(stats, ids, {})
    assert abs(h[0] - (1.0 - 0.2 + 0.4)) < 1e-9          # discharge added back, missing charge counts as 0
    assert abs(h[3600] - (1.0 - 0.5 - 0.2 - 0.3)) < 1e-9  # charge subtracted, missing discharge counts as 0
    assert 7200 not in h                                  # heat pump missing -> hour skipped
    assert 10800 not in h                                 # counter glitch (900 kWh in an hour) dropped
    # the oldest 5-minute row reports the whole counter as its change once its predecessor is purged
    five = {"imp": [dict(start=0.0, change=2000.0), dict(start=300000.0, change=0.1)]}
    assert HP.house_changes(five, {"import": "imp"}, {}, period_s=300, drop_first=True) == {300.0: 0.1}
    # without the battery options the result is the classic import - export + PV - heat pump
    h2 = HP.house_changes(stats, {k: v for k, v in ids.items() if not k.startswith("battery")}, {})
    assert abs(h2[0] - 0.8) < 1e-9


def test_house_profile_weekday_and_quarter():
    from heat_planner.engine import profile as HP
    tz = ZoneInfo(TZ)
    now = dt.datetime(2026, 11, 13, 16, 0, tzinfo=tz).timestamp()      # a Friday
    hourly, five = {}, {}
    d = dt.date(2026, 9, 1)
    while d < dt.date(2026, 11, 14):
        for h in range(24):
            s = dt.datetime(d.year, d.month, d.day, h, tzinfo=tz).timestamp()
            kw = 0.4 + (1.6 if h == 18 and d.weekday() == 1 else 0.6 if h == 18 else 0.0)   # Tuesday cooks more
            hourly[s] = kw
            for m in range(12):     # all the 18:00 energy in the first quarter
                q = m // 3
                five[s + m * 300] = (kw * 4 if (h == 18 and q == 0) else (0.0 if h == 18 else kw)) / 12
        d += dt.timedelta(days=1)
    arch = HP.merge_archive({"2020-01-01": [0.0] * 96}, HP.quarter_days(five, tz, HP._midnight(now, tz)), now, tz)
    assert "2020-01-01" not in arch and "2026-11-13" not in arch and len(arch) == (dt.date(2026, 11, 13) - dt.date(2026, 9, 1)).days
    prof = HP.build(hourly, arch, now, tz)
    assert HP.valid(prof)
    mon, tue = prof["days"][0], prof["days"][1]
    assert tue[72] > mon[72] + 0.5                           # weekday level kept (shrunk, not averaged away)
    assert mon[72] > 2 * mon[73]                             # quarter shape: peak in the first quarter of 18:00
    assert abs(sum(mon[72:76]) / 4 - 1.0) < 0.02 and abs(mon[0] - 0.4) < 1e-6   # the shape keeps the hourly energy
    t = np.array([dt.datetime(2026, 11, 17, 18, 0, tzinfo=tz).timestamp()])   # next Tuesday 18:00
    assert abs(HP.series(prof, t, tz)[0] - tue[72]) < 1e-9
    s = HP.summary(prof, now, tz)
    assert len(s["today"]) == 96 and s["evening_peak"]["tue"]["time"] == "18:00"


def test_plan_exposes_loads_and_profile():
    raw = synthetic_raw()
    o = dict(OPTIONS, **{P.PLAN_BATTERY: True, P.DHW_MODE: "plan"})
    r = planner.run(raw, o)
    nk = len(r["battery"]["code"].split("|")[1])
    assert len(r["house_kw"]) == len(r["hp_kw"]) == len(r["dhw_kw"]) == nk == len(r["tr"]) - 1
    assert all(h >= d - 1e-9 for h, d in zip(r["hp_kw"], r["dhw_kw"])) and max(r["dhw_kw"]) > 0   # hp includes hot water
    assert abs(sum(r["house_kw"]) / 4 - r["energy"]["house_kwh"] + sum(r["dhw_kw"]) / 4) < 0.2
    assert r["house_profile"]["source"] == "run" and len(r["house_profile"]["today"]) == 96
    assert abs(r["house_kw"][0] - 0.5) < 1e-6              # synthetic house: 0.6 import - 0.1 heat pump
    # a profile prepared by the coordinator is used as is
    prof = dict(planner.build_house_profile(raw, o)[0])
    prof["days"] = [[1.25] * 96 for _ in range(7)]
    r2 = planner.run(dict(raw, house_profile=prof), o)
    assert set(r2["house_kw"]) == {1.25} and r2["house_profile"]["source"] == "statistics"


def test_accounting_battery_counters():
    raw = synthetic_raw()
    tz = ZoneInfo(TZ)
    now = raw["now"]
    d0 = dt.datetime.fromtimestamp(now, tz).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    t5 = np.arange(d0 - 300, now // 900 * 900, 300)
    hod = np.array([dt.datetime.fromtimestamp(x, tz).hour for x in t5])
    rows = lambda a, key="change": [dict(start=float(x) * 1000, **{key: float(v)}) for x, v in zip(t5, a)]
    imp = np.full(len(t5), 0.05)
    bc = np.where(hod < 6, 0.1, 0.0)            # charging at night from the grid
    bd = np.where(hod >= 10, 0.02, 0.0)
    raw["stats_today"] = {"sensor.grid_import": rows(imp + bc - bd), "sensor.grid_export": rows(np.zeros(len(t5))),
                          "sensor.pv_energy": rows(np.zeros(len(t5))), "sensor.heat_pump_energy": rows(np.zeros(len(t5))),
                          "sensor.battery_energy": rows(np.full(len(t5), 3.0), "mean"),
                          "sensor.bat_in": rows(bc), "sensor.bat_out": rows(bd)}
    o = dict(OPTIONS, **{P.BAT_CHARGE_ENERGY_ENTITY: "sensor.bat_in", P.BAT_DISCHARGE_ENERGY_ENTITY: "sensor.bat_out"})
    raw["states"]["sensor.bat_in"] = dict(s="1", a={"unit_of_measurement": "kWh"})
    raw["states"]["sensor.bat_out"] = dict(s="1", a={"unit_of_measurement": "kWh"})
    td = planner.run(raw, o)["today"]
    sel = t5 >= d0
    assert td["realised_battery_source"] == "energy_counters"
    assert abs(td["realised"]["house_kwh"] - imp[sel].sum()) < 0.06            # the battery is not household load
    assert abs(td["realised"]["battery_charge_kwh"] - bc[sel].sum()) < 0.06
    # without the options: state-of-charge method as before (flat SoC -> the battery shows up in the house residual)
    td2 = planner.run(raw, dict(OPTIONS))["today"]
    assert td2["realised_battery_source"] == "state_of_charge"
    assert abs(td2["realised"]["house_kwh"] - (imp + bc - bd)[sel].sum()) < 0.06


def test_battery_soc_grid():
    from heat_planner.engine import optimizer as O
    for lo, hi, cap in ((0.142, 7.1, 7.1), (0.213, 6.39, 7.1), (0.0, 7.0, 7.0), (1.0, 1.04, 7.0)):
        bat = dict(cap=cap, soc_min=lo, soc_max=hi, ds=O.SOC_STEP)
        g, ds = O.soc_grid(bat)
        assert ds <= O.SOC_STEP + 1e-12 and abs(np.diff(g) - ds).max() < 1e-9
        assert np.abs(g - lo).min() < 1e-9                                  # the minimum charge is a level
        assert hi - lo < O.SOC_STEP or np.abs(g - hi).min() < 1e-9          # and the maximum (unless the band is narrow)
        assert g[0] >= -1e-9 and g[-1] <= cap + 1e-9 and g[0] < ds and cap - g[-1] < ds
    # the planned battery power never exceeds the limits (AC side)
    o = dict(OPTIONS, **{P.PLAN_BATTERY: True, P.BAT_CHARGE_KW: 2.4, P.BAT_DISCHARGE_KW: 2.4, P.BAT_CAPACITY: 7.1, P.BAT_MIN_SOC: 2.0})
    r = planner.run(synthetic_raw(), o)
    soc = np.array(r["battery"]["soc"])
    d = np.diff(soc) / 0.25
    assert d.max() <= 2.4 * 0.922 + 1e-6 and -d.max() <= 2.4 / 0.922 + 1e-6 and -d.min() <= 2.4 / 0.922 + 1e-6
    assert soc[1:].min() >= 7.1 * 0.02 - 0.006           # soc is published rounded to 0.01 kWh


def test_live_battery_first_quarter_limit():
    """planning limits from settings (input_number in W) for all quarters, the live (BMS) limit only for the current quarter"""
    tz = ZoneInfo(TZ)
    now = dt.datetime(2026, 11, 3, 17, 7, tzinfo=tz).timestamp()     # expensive evening: a full battery discharges at once
    raw = synthetic_raw(now)
    st = raw["states"]
    st["sensor.battery_energy"] = dict(s="6.0", a=dict(unit_of_measurement="kWh"))
    st["input_number.max_charge"] = dict(s="2400", a=dict(unit_of_measurement="W"))
    st["input_number.max_discharge"] = dict(s="2400.0", a=dict(unit_of_measurement="W"))
    st["number.power"] = dict(s="0", a=dict(unit_of_measurement="W", max_charge_power=2500, max_discharge_power=0))
    o = dict(OPTIONS, **{P.PLAN_BATTERY: True, P.BAT_CAPACITY: 7.1, P.BAT_CHARGE_KW: 3.5, P.BAT_DISCHARGE_KW: 3.5,
                         P.BAT_CHARGE_POWER_ENTITY: "input_number.max_charge", P.BAT_DISCHARGE_POWER_ENTITY: "input_number.max_discharge"})
    # planning limits read from the settings (state, no attribute, W -> kW)
    p = P.normalise(o)
    used = planner.live_battery(p, st)
    assert abs(p[P.BAT_CHARGE_KW] - 2.4) < 1e-9 and abs(p[P.BAT_DISCHARGE_KW] - 2.4) < 1e-9
    assert used[P.BAT_DISCHARGE_KW] == "input_number.max_discharge" and p["_live_discharge_kw"] is None
    base = planner.run(raw, o)["battery"]
    soc = np.array(base["soc"])
    assert soc[0] - soc[1] > 0.05, base["code"][:20]                    # without a live limit slot 0 discharges
    assert base["live_charge_kw"] is None and base["live_discharge_kw"] is None
    lo = dict(o, **{P.BAT_LIVE_CHARGE_POWER_ENTITY: "number.power", P.BAT_LIVE_CHARGE_POWER_ATTR: "max_charge_power",
                    P.BAT_LIVE_DISCHARGE_POWER_ENTITY: "number.power", P.BAT_LIVE_DISCHARGE_POWER_ATTR: "max_discharge_power"})
    p = P.normalise(lo)
    used = planner.live_battery(p, st)
    assert p["_live_discharge_kw"] == 0.0 and abs(p["_live_charge_kw"] - 2.4) < 1e-9   # min(planning limit, live value)
    assert used["live_discharge_kw"] == "number.power.max_discharge_power"
    b = planner.run(raw, lo)["battery"]
    soc = np.array(b["soc"])
    assert soc[1] >= soc[0] - 1e-9 and b["code"].split("|")[1][0] not in "DX"   # live 0 W discharge in the current quarter
    assert soc[1] - soc[2] > 0.05                                      # the next quarter discharges again (planning limit)
    d = np.diff(soc) / 0.25
    assert d.max() <= 2.4 * 0.922 + 1e-6 and -d.min() <= 2.4 / 0.922 + 1e-6
    assert b["live_discharge_kw"] == 0.0 and b["discharge_kw"] == 2.4
    # live entity unavailable -> no first-quarter limit (backwards compatible)
    st["number.power"] = dict(s="unavailable", a=dict(unit_of_measurement="W"))
    p = P.normalise(lo)
    planner.live_battery(p, st)
    assert p["_live_charge_kw"] is None and p["_live_discharge_kw"] is None


def test_empty_battery_plans_solar_charge_and_evening_discharge():
    """battery at its minimum with a live discharge limit of 0 W: charge from solar at midday, discharge in the evening"""
    tz = ZoneInfo(TZ)
    now = dt.datetime(2026, 11, 3, 0, 7, tzinfo=tz).timestamp()
    raw = synthetic_raw(now)
    st = raw["states"]
    st["sensor.battery_energy"] = dict(s="2", a=dict(unit_of_measurement="%"))
    st["number.power"] = dict(s="0", a=dict(unit_of_measurement="W", max_charge_power=2500, max_discharge_power=0))
    o = dict(OPTIONS, **{P.PLAN_BATTERY: True, P.BAT_CAPACITY: 7.1, P.BAT_MIN_SOC: 2.0, P.BAT_CHARGE_KW: 2.4, P.BAT_DISCHARGE_KW: 2.4,
                         P.BAT_LIVE_CHARGE_POWER_ENTITY: "number.power", P.BAT_LIVE_CHARGE_POWER_ATTR: "max_charge_power",
                         P.BAT_LIVE_DISCHARGE_POWER_ENTITY: "number.power", P.BAT_LIVE_DISCHARGE_POWER_ATTR: "max_discharge_power"})
    b = planner.run(raw, o)["battery"]
    code = b["code"].split("|")[1]
    hour = lambda i: dt.datetime.fromtimestamp(planner.I.q0(now) + i * 900, tz).hour
    assert any(c == "C" and 9 <= hour(i) < 16 for i, c in enumerate(code)), code
    assert any(c in "DX" and 17 <= hour(i) < 23 for i, c in enumerate(code)), code
    d = np.diff(np.array(b["soc"])) / 0.25
    assert d.max() <= 2.4 * 0.922 + 1e-6 and -d.min() <= 2.4 / 0.922 + 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
