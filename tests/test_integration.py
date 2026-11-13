"""Home Assistant tests: config flow, options flow, YAML import and a full planning run with sensors."""
from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import pytest

pytest.importorskip("pytest_homeassistant_custom_component")

from homeassistant import config_entries  # noqa: E402
from homeassistant.core import HomeAssistant, SupportsResponse  # noqa: E402
from homeassistant.data_entry_flow import FlowResultType  # noqa: E402
from homeassistant.setup import async_setup_component  # noqa: E402
from pytest_homeassistant_custom_component.common import MockConfigEntry  # noqa: E402

from custom_components.heat_planner.const import DOMAIN  # noqa: E402
from custom_components.heat_planner.engine import params as P  # noqa: E402

from .test_engine import OPTIONS, synthetic_raw  # noqa: E402


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(recorder_mock, enable_custom_integrations):
    yield


async def test_config_flow(hass: HomeAssistant) -> None:
    r = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})
    assert r["type"] is FlowResultType.FORM and r["step_id"] == "user"
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {
        "name": "Heat Planner", P.ROOM_TEMP_ENTITY: "sensor.room_temperature", P.OUTDOOR_TEMP_ENTITY: "sensor.outdoor_temperature",
        P.WEATHER_ENTITY: "weather.home", P.FORECAST_BIAS: "0", P.HEAT_OUTPUT_ENTITY: "sensor.heat_pump_heat_power"})
    assert r["step_id"] == "energy"
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {P.IMPORT_ENERGY_ENTITY: "sensor.grid_import"})
    assert r["step_id"] == "prices"
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {
        P.PRICE_SOURCE: "tibber", P.PRICE_ENTITY: "sensor.electricity_price", P.NET_METERING: False})
    assert r["step_id"] == "devices"
    # battery planning without a state of charge sensor is rejected
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {P.PLAN_BATTERY: True, P.DHW_MODE: "off"})
    assert r["type"] is FlowResultType.FORM and r["errors"] == {P.BATTERY_SOC_ENTITY: "soc_entity_required"}
    with patch("custom_components.heat_planner.async_setup_entry", return_value=True):
        r = await hass.config_entries.flow.async_configure(r["flow_id"], {P.PLAN_BATTERY: False, P.DHW_MODE: "off"})
    assert r["type"] is FlowResultType.CREATE_ENTRY and r["title"] == "Heat Planner"
    opts = r["result"].options
    assert opts[P.ROOM_TEMP_ENTITY] == "sensor.room_temperature"
    assert opts[P.HEAT_LOSS] == P.NUMBERS[P.HEAT_LOSS][0]
    assert P.HEATING_ACTIVE_ENTITY not in opts


async def test_options_flow(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(domain=DOMAIN, title="Heat Planner", data={}, options={**P.defaults(), **OPTIONS})
    entry.add_to_hass(hass)
    with patch("custom_components.heat_planner.async_setup_entry", return_value=True):
        r = await hass.config_entries.options.async_init(entry.entry_id)
        assert r["type"] is FlowResultType.MENU
        r = await hass.config_entries.options.async_configure(r["flow_id"], {"next_step_id": "comfort"})
        assert r["step_id"] == "comfort"
        bad = {k: P.defaults()[k] for k in ("min_temp", "max_temp", "hard_min_temp", "safety_margin", "min_block_hours",
                                             "start_cost", "tail_hours", "update_minutes")}
        r2 = await hass.config_entries.options.async_configure(r["flow_id"], {**bad, P.MAX_TEMP: 18.6})
        assert r2["errors"] == {P.MAX_TEMP: "comfort_band"}
        r3 = await hass.config_entries.options.async_configure(r["flow_id"], {**bad, P.MIN_TEMP: 19.0})
        assert r3["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options[P.MIN_TEMP] == 19.0
    assert entry.options[P.ROOM_TEMP_ENTITY] == "sensor.room_temperature"
    # energy section: optional battery energy counters can be set and cleared again
    energy = {k: OPTIONS[k] for k in (P.IMPORT_ENERGY_ENTITY, P.EXPORT_ENERGY_ENTITY, P.PV_ENERGY_ENTITY, P.HP_ENERGY_ENTITY)}
    with patch("custom_components.heat_planner.async_setup_entry", return_value=True):
        r = await hass.config_entries.options.async_init(entry.entry_id)
        r = await hass.config_entries.options.async_configure(r["flow_id"], {"next_step_id": "energy"})
        assert r["step_id"] == "energy" and P.BAT_CHARGE_ENERGY_ENTITY in str(r["data_schema"].schema)
        r = await hass.config_entries.options.async_configure(r["flow_id"], {
            **energy, P.BAT_CHARGE_ENERGY_ENTITY: "sensor.bat_in", P.BAT_DISCHARGE_ENERGY_ENTITY: "sensor.bat_out"})
        assert r["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options[P.BAT_CHARGE_ENERGY_ENTITY] == "sensor.bat_in" and entry.options[P.MIN_TEMP] == 19.0
    with patch("custom_components.heat_planner.async_setup_entry", return_value=True):
        r = await hass.config_entries.options.async_init(entry.entry_id)
        r = await hass.config_entries.options.async_configure(r["flow_id"], {"next_step_id": "energy"})
        r = await hass.config_entries.options.async_configure(r["flow_id"], energy)
    assert P.BAT_CHARGE_ENERGY_ENTITY not in entry.options and P.BAT_DISCHARGE_ENERGY_ENTITY not in entry.options


async def test_yaml_import(hass: HomeAssistant) -> None:
    with patch("custom_components.heat_planner.async_setup_entry", return_value=True):
        assert await async_setup_component(hass, DOMAIN, {DOMAIN: {**OPTIONS, "name": "Imported", P.MIN_TEMP: 19.5}})
        await hass.async_block_till_done()
    entries = hass.config_entries.async_entries(DOMAIN)
    assert len(entries) == 1 and entries[0].title == "Imported" and entries[0].options[P.MIN_TEMP] == 19.5


async def test_setup_and_sensors(hass: HomeAssistant) -> None:
    """full run with the real data collection: Tibber and weather services are faked, states set in hass"""
    raw = synthetic_raw(dt.datetime.now().timestamp())
    for eid, s in raw["states"].items():
        hass.states.async_set(eid, s["s"], s["a"])

    async def prices(call):
        return {"prices": {"Home": [dict(start_time=dt.datetime.fromtimestamp(x["start"]).astimezone().isoformat(), price=x["price"])
                                    for x in raw["prices"]]}}

    async def forecasts(call):
        return {"weather.home": {"forecast": raw["weather"]}}

    hass.services.async_register("tibber", "get_prices", prices, supports_response=SupportsResponse.ONLY)
    hass.services.async_register("weather", "get_forecasts", forecasts, supports_response=SupportsResponse.ONLY)
    calls = []
    hass.bus.async_listen("call_service", lambda e: calls.append(e.data))
    entry = MockConfigEntry(domain=DOMAIN, title="Heat Planner", data={},
                            options={**P.defaults(), **OPTIONS, P.PLAN_BATTERY: True, P.DHW_MODE: "plan"})
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    plan = hass.states.get("sensor.heat_planner_plan")
    assert plan is not None and plan.attributes.get("error") is None, plan
    assert plan.attributes["created"] and plan.attributes["tr"]
    assert len(plan.attributes["house_kw"]) == len(plan.attributes["hp_kw"]) == len(plan.attributes["tr"]) - 1
    assert len(plan.attributes["house_profile"]["today"]) == 96
    bat = hass.states.get("sensor.heat_planner_battery")
    assert bat.attributes["enabled"] is True and bat.attributes["soc"]
    hw = hass.states.get("sensor.heat_planner_hot_water")
    assert hw.attributes["mode"] == "plan"
    # read-only: the only services called are the two data services
    assert {(c["domain"], c["service"]) for c in calls} <= {("tibber", "get_prices"), ("weather", "get_forecasts")}
    await hass.services.async_call(DOMAIN, "replan", {}, blocking=True)
    assert await hass.config_entries.async_unload(entry.entry_id)
