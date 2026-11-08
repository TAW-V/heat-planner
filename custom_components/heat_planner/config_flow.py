"""Config flow and options flow for Heat Planner."""
from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.core import callback
from homeassistant.helpers import selector as sel

from .const import ALL_KEYS, DEFAULT_NAME, DOMAIN, ENTITY_FIELDS, NUMBER_UI, SECTIONS, SELECTS, SETUP_STEPS
from .engine import params as P

CONF_NAME = "name"


def _field(key: str, values: dict) -> tuple[Any, Any]:
    cur = values.get(key)
    if key in ENTITY_FIELDS:
        domains, dc, multiple, required = ENTITY_FIELDS[key]
        cfg: dict[str, Any] = {"domain": domains, "multiple": multiple}
        if dc:
            cfg["device_class"] = dc
        s = sel.EntitySelector(sel.EntitySelectorConfig(**cfg))
        if required:
            return (vol.Required(key, default=cur) if cur else vol.Required(key)), s
        return vol.Optional(key, description={"suggested_value": cur}), s
    if key in P.NUMBERS:
        _, lo, hi = P.NUMBERS[key]
        unit, step = NUMBER_UI.get(key, (None, "any"))
        cfg = {"min": lo, "max": hi, "step": step, "mode": sel.NumberSelectorMode.BOX}
        if unit:
            cfg["unit_of_measurement"] = unit
        return vol.Required(key, default=cur), sel.NumberSelector(sel.NumberSelectorConfig(**cfg))
    if key in P.BOOLEANS:
        return vol.Required(key, default=bool(cur)), sel.BooleanSelector()
    if key in SELECTS:
        return vol.Required(key, default=cur), sel.SelectSelector(sel.SelectSelectorConfig(
            options=SELECTS[key], translation_key=key, mode=sel.SelectSelectorMode.DROPDOWN))
    return vol.Required(key, default=str(cur if cur is not None else "")), sel.TextSelector()


def build_schema(keys: list[str], values: dict, extra: dict | None = None) -> vol.Schema:
    fields = dict(extra or {})
    for k in keys:
        marker, s = _field(k, values)
        fields[marker] = s
    return vol.Schema(fields)


def merge(options: dict, keys: list[str], user_input: dict) -> dict:
    """apply one form: optional entity fields that were cleared are removed"""
    out = dict(options)
    for k in keys:
        if k in user_input and user_input[k] not in (None, "", []):
            out[k] = user_input[k]
        elif k in ENTITY_FIELDS:
            out.pop(k, None)
    return out


def validate(opts: dict) -> dict[str, str]:
    errors: dict[str, str] = {}
    try:
        P.parse_bias(opts.get(P.FORECAST_BIAS, "0"))
    except (TypeError, ValueError):
        errors[P.FORECAST_BIAS] = "invalid_bias"
    d = P.defaults()
    d.update(opts)
    if float(d[P.MAX_TEMP]) < float(d[P.MIN_TEMP]) + 0.5:
        errors[P.MAX_TEMP] = "comfort_band"
    if float(d[P.HARD_MIN_TEMP]) > float(d[P.MIN_TEMP]):
        errors[P.HARD_MIN_TEMP] = "hard_min"
    if float(d[P.BAT_MIN_SOC]) > float(d[P.BAT_MAX_SOC]):
        errors[P.BAT_MAX_SOC] = "soc_range"
    if d.get(P.DHW_MODE) == "schedule" and not d.get(P.DHW_SCHEDULE_ENTITY):
        errors[P.DHW_SCHEDULE_ENTITY] = "schedule_entity_required"
    if d.get(P.PLAN_BATTERY) and not d.get(P.BATTERY_SOC_ENTITY):
        errors[P.BATTERY_SOC_ENTITY] = "soc_entity_required"
    return errors


def clean(opts: dict) -> dict:
    return {k: v for k, v in opts.items() if k in ALL_KEYS and v is not None}


class HeatPlannerConfigFlow(ConfigFlow, domain=DOMAIN):
    """initial setup: sources, energy, prices, battery / hot water; all parameters start at their defaults"""

    VERSION = 1

    def __init__(self) -> None:
        self._opts: dict[str, Any] = P.defaults()
        self._name = DEFAULT_NAME

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return HeatPlannerOptionsFlow()

    async def _step(self, step: str, nxt, user_input, extra_schema=None):
        keys = SETUP_STEPS[step]
        errors: dict[str, str] = {}
        if user_input is not None:
            if CONF_NAME in user_input:
                self._name = user_input.pop(CONF_NAME) or DEFAULT_NAME
            opts = merge(self._opts, keys, user_input)
            errors = {k: v for k, v in validate(opts).items() if k in keys}
            if not errors:
                self._opts = opts
                return await nxt()
        return self.async_show_form(step_id=step, data_schema=build_schema(keys, self._opts, extra_schema), errors=errors)

    async def async_step_user(self, user_input=None) -> ConfigFlowResult:
        extra = {vol.Required(CONF_NAME, default=self._name): sel.TextSelector()}
        return await self._step("user", self.async_step_energy, user_input, extra)

    async def async_step_energy(self, user_input=None) -> ConfigFlowResult:
        return await self._step("energy", self.async_step_prices, user_input)

    async def async_step_prices(self, user_input=None) -> ConfigFlowResult:
        return await self._step("prices", self.async_step_devices, user_input)

    async def async_step_devices(self, user_input=None) -> ConfigFlowResult:
        return await self._step("devices", self._finish, user_input)

    async def _finish(self) -> ConfigFlowResult:
        return self.async_create_entry(title=self._name, data={}, options=clean(self._opts))

    async def async_step_import(self, import_data: dict[str, Any]) -> ConfigFlowResult:
        """one-time import of a `heat_planner:` YAML block (migration helper)"""
        await self.async_set_unique_id("yaml_import")
        self._abort_if_unique_id_configured()
        data = dict(import_data)
        name = data.pop(CONF_NAME, DEFAULT_NAME)
        opts = P.defaults()
        opts.update(data)
        return self.async_create_entry(title=name, data={}, options=clean(opts))


class HeatPlannerOptionsFlow(OptionsFlow):
    """all settings, grouped in a menu"""

    async def async_step_init(self, user_input=None) -> ConfigFlowResult:
        return self.async_show_menu(step_id="init", menu_options=list(SECTIONS))

    async def _section(self, step: str, user_input):
        keys = SECTIONS[step]
        current = P.defaults()
        current.update(self.config_entry.options)
        errors: dict[str, str] = {}
        if user_input is not None:
            opts = merge(current, keys, user_input)
            errors = {k: v for k, v in validate(opts).items() if k in keys}
            if not errors:
                return self.async_create_entry(data=clean(opts))
            current = opts
        return self.async_show_form(step_id=step, data_schema=build_schema(keys, current), errors=errors)

    async def async_step_sources(self, user_input=None) -> ConfigFlowResult:
        return await self._section("sources", user_input)

    async def async_step_energy(self, user_input=None) -> ConfigFlowResult:
        return await self._section("energy", user_input)

    async def async_step_prices(self, user_input=None) -> ConfigFlowResult:
        return await self._section("prices", user_input)

    async def async_step_comfort(self, user_input=None) -> ConfigFlowResult:
        return await self._section("comfort", user_input)

    async def async_step_building(self, user_input=None) -> ConfigFlowResult:
        return await self._section("building", user_input)

    async def async_step_heat_pump(self, user_input=None) -> ConfigFlowResult:
        return await self._section("heat_pump", user_input)

    async def async_step_battery(self, user_input=None) -> ConfigFlowResult:
        return await self._section("battery", user_input)

    async def async_step_hot_water(self, user_input=None) -> ConfigFlowResult:
        return await self._section("hot_water", user_input)
