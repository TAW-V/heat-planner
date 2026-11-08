"""Plan sensors (read-only)."""
from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import slugify

from .const import DOMAIN
from .coordinator import HeatPlannerCoordinator

# per-slot curves and price lists: visible in the UI, not written to the recorder
PLAN_UNRECORDED = frozenset({"tr", "ts", "tr_tail", "state_estimate", "blocks_without_battery", "energy", "cost_breakdown"})
BATTERY_UNRECORDED = frozenset({"soc", "prices", "feed_in_prices", "blocks", "code"})
DHW_UNRECORDED = frozenset({"blocks"})


async def async_setup_entry(hass: HomeAssistant, entry, async_add_entities: AddEntitiesCallback) -> None:
    coordinator: HeatPlannerCoordinator = entry.runtime_data
    async_add_entities([PlanSensor(coordinator, entry), BatterySensor(coordinator, entry), HotWaterSensor(coordinator, entry)])


class _Base(CoordinatorEntity[HeatPlannerCoordinator], SensorEntity):
    _attr_has_entity_name = True
    key = ""

    def __init__(self, coordinator: HeatPlannerCoordinator, entry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_{self.key}"
        self._attr_translation_key = self.key
        self.entity_id = f"sensor.{slugify(entry.title)}_{self.key}"
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry.entry_id)}, name=entry.title,
                                            entry_type=DeviceEntryType.SERVICE, manufacturer="Heat Planner",
                                            model="Read-only planner")
        self._update_from(coordinator)

    @property
    def available(self) -> bool:
        return True

    @callback
    def _handle_coordinator_update(self) -> None:
        self._update_from(self.coordinator)
        self.async_write_ha_state()

    def _update_from(self, c: HeatPlannerCoordinator) -> None:
        raise NotImplementedError


class PlanSensor(_Base):
    """space-heating plan: state = heating blocks 'ISO-start/hours;...' (known prices), attributes = details"""

    key = "plan"
    _attr_icon = "mdi:radiator"
    _unrecorded_attributes = PLAN_UNRECORDED

    def _update_from(self, c: HeatPlannerCoordinator) -> None:
        r = c.result or {}
        self._attr_native_value = ((r.get("schedule") or "none")[:255]) if r else None
        attrs: dict[str, Any] = {k: v for k, v in r.items() if k not in ("battery", "hot_water") and not k.startswith("_")}
        attrs["error"] = c.error
        self._attr_extra_state_attributes = attrs


class BatterySensor(_Base):
    """battery plan: state = readable summary, attributes = blocks, SoC curve and per-quarter code"""

    key = "battery"
    _attr_icon = "mdi:battery-clock"
    _unrecorded_attributes = BATTERY_UNRECORDED

    def _update_from(self, c: HeatPlannerCoordinator) -> None:
        r = c.result or {}
        b = r.get("battery")
        if not r:
            self._attr_native_value = None
        elif b is None:
            self._attr_native_value = "off"
        else:
            self._attr_native_value = (b.get("summary") or "")[:255] or None
        self._attr_extra_state_attributes = {"created": r.get("created"), "enabled": b is not None,
                                             "net_metering": r.get("net_metering"), **(b or {}), "error": c.error}


class HotWaterSensor(_Base):
    """hot water plan: state = readable summary, attributes = planned charges"""

    key = "hot_water"
    _attr_icon = "mdi:water-boiler"
    _unrecorded_attributes = DHW_UNRECORDED

    def _update_from(self, c: HeatPlannerCoordinator) -> None:
        r = c.result or {}
        h = r.get("hot_water")
        self._attr_native_value = ((h.get("summary") or "")[:255] or None) if h else None
        self._attr_extra_state_attributes = {"created": r.get("created"), **(h or {}), "error": c.error}
