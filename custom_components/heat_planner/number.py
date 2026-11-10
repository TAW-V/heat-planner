"""Comfort settings as number entities, so they can be changed from a dashboard.

Each entity reads and writes the matching option of the config entry. Changing one of these options does not reload
the integration; the planner simply runs again with the new value.
"""
from __future__ import annotations

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.const import EntityCategory, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import slugify

from .const import DOMAIN, LIVE_NUMBERS, NUMBER_UI
from .engine import params as P

ICONS = {P.MIN_TEMP: "mdi:thermometer-low", P.MAX_TEMP: "mdi:thermometer-high", P.HARD_MIN_TEMP: "mdi:thermometer-alert",
         P.SAFETY_MARGIN: "mdi:shield-half-full"}


async def async_setup_entry(hass: HomeAssistant, entry, async_add_entities: AddEntitiesCallback) -> None:
    async_add_entities([OptionNumber(hass, entry, key) for key in LIVE_NUMBERS])


class OptionNumber(NumberEntity):
    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_mode = NumberMode.BOX
    _attr_entity_category = EntityCategory.CONFIG

    def __init__(self, hass: HomeAssistant, entry, key: str) -> None:
        self.hass = hass
        self._entry = entry
        self._key = key
        _, lo, hi = P.NUMBERS[key]
        unit, step = NUMBER_UI.get(key, (None, 0.1))
        self._attr_native_min_value = lo
        self._attr_native_max_value = hi
        self._attr_native_step = step if isinstance(step, (int, float)) else 0.1
        self._attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS if unit == "°C" else unit
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_translation_key = key
        self._attr_icon = ICONS.get(key)
        self.entity_id = f"number.{slugify(entry.title)}_{key}"
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry.entry_id)}, name=entry.title,
                                            entry_type=DeviceEntryType.SERVICE, manufacturer="Heat Planner",
                                            model="Read-only planner")

    async def async_added_to_hass(self) -> None:
        self.async_on_remove(self._entry.add_update_listener(self._options_changed))

    async def _options_changed(self, hass: HomeAssistant, entry) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> float:
        v = self._entry.options.get(self._key)
        return float(P.defaults()[self._key] if v is None else v)

    async def async_set_native_value(self, value: float) -> None:
        from .config_flow import validate   # local import: config_flow imports this package's constants

        opts = dict(self._entry.options)
        opts[self._key] = round(float(value), 2)
        errors = validate(opts)
        if errors:
            raise ServiceValidationError(
                f"{self._key}={value}: invalid ({', '.join(f'{k}: {v}' for k, v in errors.items())}); "
                "keep max >= min + 0.5 and hard minimum <= min")
        self.hass.config_entries.async_update_entry(self._entry, options=opts)
