"""Data collection and planning runs.

Read-only by design: this module only reads states, statistics and forecasts and calls services that return
data (`tibber.get_prices`, `weather.get_forecasts`). It never calls a service that changes a device or helper.
"""
from __future__ import annotations

import datetime as dt
import importlib
import logging
from typing import Any

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.statistics import statistics_during_period
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .engine import params as P

_LOGGER = logging.getLogger(__name__)

HOURLY_DAYS = 8
FIVE_MIN_DAYS = 2
PROFILE_SHORT_DAYS = 11      # 5-minute counters fetched for the quarter shape (the recorder keeps ~10 days)
STORE_VERSION = 1


def _rows(stats: dict) -> dict:
    """recorder rows (start = datetime or seconds) -> plain rows with start in milliseconds"""
    out = {}
    for k, v in stats.items():
        out[k] = [dict(r, start=(r["start"].timestamp() if hasattr(r["start"], "timestamp") else float(r["start"])) * 1000)
                  for r in v]
    return out


class HeatPlannerCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """runs the planner every `update_minutes` and keeps the last good plan when a run fails"""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.entry = entry
        self.options = dict(entry.options)
        minutes = P.normalise(self.options)[P.UPDATE_MINUTES]
        super().__init__(hass, _LOGGER, name=f"{DOMAIN} {entry.title}", update_interval=dt.timedelta(minutes=minutes),
                         config_entry=entry)
        self._planner = None
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        # household profile: rebuilt once a day; quarter values of complete days are archived in .storage
        self._store: Store = Store(hass, STORE_VERSION, f"{DOMAIN}.{entry.entry_id}.house_quarters")
        self._archive: dict | None = None
        self._profile: dict | None = None
        self._profile_day: str | None = None

    def _opt(self, key: str):
        return self.options.get(key)

    async def _collect(self) -> dict[str, Any]:
        hass = self.hass
        now = dt_util.now()
        o = self.options
        ids = [o.get(k) for k in P.ENTITY_KEYS if o.get(k)] + [e for k in P.ENTITY_LIST_KEYS for e in (o.get(k) or [])]
        states = {}
        for eid in ids:
            s = hass.states.get(eid)
            if s is not None:
                states[eid] = dict(s=s.state, a=dict(s.attributes), lc=s.last_changed.timestamp())
        prices: list[dict] = []
        if o.get(P.PRICE_SOURCE, "tibber") == "tibber":
            d0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
            resp = await hass.services.async_call(
                "tibber", "get_prices",
                {"start": d0.strftime("%Y-%m-%d %H:%M:%S"), "end": (d0 + dt.timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")},
                blocking=True, return_response=True)
            homes = list(((resp or {}).get("prices") or {}).values())
            if homes:
                prices = [dict(start=x["start_time"], price=x["price"]) for x in homes[0]]
        else:
            eng = self._planner
            pe = o.get(P.PRICE_ENTITY)
            st = states.get(pe) or {}
            prices = eng.I.prices_from_attributes(st.get("a"), (st.get("a") or {}).get("unit_of_measurement"))
        weather: list[dict] = []
        wid = o.get(P.WEATHER_ENTITY)
        if wid:
            try:
                wf = await hass.services.async_call("weather", "get_forecasts", {"type": "hourly"}, target={"entity_id": wid},
                                                    blocking=True, return_response=True)
                weather = list(((wf or {}).get(wid) or {}).get("forecast") or [])
            except Exception as err:  # noqa: BLE001 - a missing forecast is handled by the engine
                _LOGGER.warning("weather forecast unavailable: %s", err)
        rec = get_instance(hass)
        hour_ids = {o.get(k) for k in P.ENERGY_ROLES.values() if o.get(k)}
        start_h = (now - dt.timedelta(days=HOURLY_DAYS)).replace(minute=0, second=0, microsecond=0)
        stats_hour = {}
        if hour_ids:
            stats_hour.update(await rec.async_add_executor_job(
                statistics_during_period, hass, start_h, None, hour_ids, "hour", None, {"change"}))
        if o.get(P.PRICE_ENTITY):
            stats_hour.update(await rec.async_add_executor_job(
                statistics_during_period, hass, start_h, None, {o[P.PRICE_ENTITY]}, "hour", None, {"mean"}))
        five_ids = {o.get(k) for k in (P.ROOM_TEMP_ENTITY, P.OUTDOOR_TEMP_ENTITY, P.HEAT_OUTPUT_ENTITY, P.HEATING_ACTIVE_ENTITY,
                                       P.PV_POWER_ENTITY, P.DHW_TANK_ENTITY) if o.get(k)}
        stats_5 = await rec.async_add_executor_job(
            statistics_during_period, hass, now - dt.timedelta(days=FIVE_MIN_DAYS), None, five_ids, "5minute", None, {"mean", "max"})
        # today since local midnight (5-minute counters + battery level) for the day accounting
        today_ids = {o.get(k) for k in (*P.ENERGY_ROLES.values(), P.BATTERY_SOC_ENTITY) if o.get(k)}
        stats_today = {}
        if today_ids:
            try:
                start_d = now.replace(hour=0, minute=0, second=0, microsecond=0) - dt.timedelta(minutes=5)
                stats_today = await rec.async_add_executor_job(
                    statistics_during_period, hass, start_d, None, today_ids, "5minute", None, {"change", "mean"})
            except Exception as err:  # noqa: BLE001 - the day accounting is optional
                _LOGGER.debug("statistics for today unavailable: %s", err)
        raw = dict(now=now.timestamp(), tz=str(hass.config.time_zone), states=states, prices=prices, weather=weather,
                   stats_hour=_rows(stats_hour), stats_5min=_rows(stats_5), stats_today=_rows(stats_today))
        profile = await self._house_profile(now, hour_ids, states)
        if profile is not None:
            raw["house_profile"] = profile
        return raw

    async def _house_profile(self, now: dt.datetime, ids: set, states: dict) -> dict | None:
        """household profile per weekday and quarter, rebuilt on the first run of each day (statistics of many weeks);
        None when it cannot be built (the engine then builds a simpler one from this run's statistics)"""
        day = now.date().isoformat()
        if self._profile is not None and self._profile_day == day:
            return self._profile
        if not ids:
            return None
        try:
            if self._archive is None:
                stored = await self._store.async_load() or {}
                self._archive = stored.get("days") or {}
            rec = get_instance(self.hass)
            eng = self._planner
            weeks = eng.HP.WEEKS
            d0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
            hourly = await rec.async_add_executor_job(
                statistics_during_period, self.hass, d0 - dt.timedelta(days=weeks * 7 + 1), None, ids, "hour", None, {"change"})
            five = await rec.async_add_executor_job(
                statistics_during_period, self.hass, now - dt.timedelta(days=PROFILE_SHORT_DAYS), None, ids, "5minute", None,
                {"change"})
            raw = dict(now=now.timestamp(), tz=str(self.hass.config.time_zone), states=states, stats_hour=_rows(hourly),
                       stats_5min_energy=_rows(five))
            profile, archive = await self.hass.async_add_executor_job(eng.build_house_profile, raw, self.options, self._archive)
        except Exception as err:  # noqa: BLE001 - the planner falls back to the statistics of the run
            _LOGGER.warning("household profile not built: %s", err)
            return None
        if not eng.HP.valid(profile):      # no statistics yet: let the engine use the statistics of each run
            return None
        if archive != self._archive:
            self._archive = archive
            self._store.async_delay_save(lambda: {"days": self._archive}, 60)
        self._profile, self._profile_day = profile, day
        return profile

    async def _async_update_data(self) -> dict[str, Any]:
        if self._planner is None:
            # numpy import and module loading are blocking: do it in the executor
            self._planner = await self.hass.async_add_import_executor_job(
                importlib.import_module, f"{__package__}.engine.planner")
        lang = "nl" if str(self.hass.config.language or "").lower().startswith("nl") else "en"
        try:
            raw = await self._collect()
            self.result = await self.hass.async_add_executor_job(self._planner.run, raw, self.options, lang, self.result)
            self.error = None
        except Exception as err:  # noqa: BLE001 - keep the previous plan and report the error on the sensors
            _LOGGER.exception("planning run failed")
            self.error = f"{type(err).__name__}: {err}"
        return {"result": self.result, "error": self.error}
