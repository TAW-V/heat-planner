"""Heat Planner: predictive, read-only planner for heat pump space heating, hot water and a home battery.

The integration computes plans and publishes them as sensors. It never switches, sets or otherwise controls a
device; use the sensors in your own automations if you want to act on the plan.
"""
from __future__ import annotations

import logging

import voluptuous as vol

from homeassistant.config_entries import SOURCE_IMPORT, ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import CoreState, HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN, PLATFORMS, SERVICE_REPLAN
from .coordinator import HeatPlannerCoordinator

_LOGGER = logging.getLogger(__name__)

# optional one-time YAML import (all keys as in the options); the UI is the normal way to configure
CONFIG_SCHEMA = vol.Schema({vol.Optional(DOMAIN): vol.Schema({cv.string: object})}, extra=vol.ALLOW_EXTRA)

HeatPlannerConfigEntry = ConfigEntry[HeatPlannerCoordinator]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    async def _replan(call: ServiceCall) -> None:
        for entry in hass.config_entries.async_loaded_entries(DOMAIN):
            await entry.runtime_data.async_refresh()

    hass.services.async_register(DOMAIN, SERVICE_REPLAN, _replan, schema=vol.Schema({}))
    if DOMAIN in config:
        hass.async_create_task(hass.config_entries.flow.async_init(
            DOMAIN, context={"source": SOURCE_IMPORT}, data=dict(config[DOMAIN])))
    return True


async def async_setup_entry(hass: HomeAssistant, entry: HeatPlannerConfigEntry) -> bool:
    coordinator = HeatPlannerCoordinator(hass, entry)
    entry.runtime_data = coordinator
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_reload))

    pending = {}

    async def _first(_event=None) -> None:
        pending.pop("unsub", None)
        await coordinator.async_refresh()

    def _cancel() -> None:
        if (unsub := pending.pop("unsub", None)) is not None:
            unsub()

    # the price and weather integrations must be loaded before the first run
    if hass.state is CoreState.running:
        entry.async_create_background_task(hass, _first(), f"{DOMAIN}_first_refresh")
    else:
        pending["unsub"] = hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _first)
        entry.async_on_unload(_cancel)
    return True


async def _reload(hass: HomeAssistant, entry: HeatPlannerConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: HeatPlannerConfigEntry) -> bool:
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
