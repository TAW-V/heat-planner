"""Constants for the Heat Planner integration."""
from __future__ import annotations

from .engine import params as P

DOMAIN = "heat_planner"
DEFAULT_NAME = "Heat Planner"
PLATFORMS = ["sensor"]
SERVICE_REPLAN = "replan"

# ---------------------------------------------------------------------------------------------- option groups (UI)
# entity fields: key -> (selector domain(s), device_class or None, multiple, required)
ENTITY_FIELDS: dict[str, tuple[list[str], str | None, bool, bool]] = {
    P.ROOM_TEMP_ENTITY: (["sensor"], "temperature", False, True),
    P.OUTDOOR_TEMP_ENTITY: (["sensor"], "temperature", False, True),
    P.WEATHER_ENTITY: (["weather"], None, False, True),
    P.HEAT_OUTPUT_ENTITY: (["sensor"], "power", False, False),
    P.HEATING_ACTIVE_ENTITY: (["sensor", "binary_sensor"], None, False, False),
    P.IMPORT_ENERGY_ENTITY: (["sensor"], "energy", False, True),
    P.EXPORT_ENERGY_ENTITY: (["sensor"], "energy", False, False),
    P.PV_ENERGY_ENTITY: (["sensor"], "energy", False, False),
    P.HP_ENERGY_ENTITY: (["sensor"], "energy", False, False),
    P.PV_POWER_ENTITY: (["sensor"], "power", False, False),
    P.PV_FORECAST_ENTITIES: (["sensor"], None, True, False),
    P.PRICE_ENTITY: (["sensor"], None, False, True),
    P.PRICE_BREAKDOWN_ENTITY: (["sensor"], None, False, False),
    P.PEAK_ENTITY: (["binary_sensor", "sensor"], None, False, False),
    P.NET_METERING_ENTITY: (["input_boolean", "switch", "binary_sensor"], None, False, False),
    P.EXPORT_PRICE_ENTITY: (["sensor"], None, False, False),
    P.BATTERY_SOC_ENTITY: (["sensor"], None, False, False),
    P.DHW_SCHEDULE_ENTITY: (["input_text", "sensor", "text"], None, False, False),
    P.DHW_TANK_ENTITY: (["sensor"], "temperature", False, False),
    P.LEGIONELLA_ENTITY: (["script", "switch", "input_boolean", "binary_sensor"], None, False, False),
}

# number fields: key -> (unit, step)
NUMBER_UI: dict[str, tuple[str | None, float | str]] = {
    P.MIN_TEMP: ("°C", 0.1), P.MAX_TEMP: ("°C", 0.1), P.HARD_MIN_TEMP: ("°C", 0.1), P.MIN_BLOCK_HOURS: ("h", 0.25),
    P.START_COST: (None, 0.01), P.SAFETY_MARGIN: ("K", 0.1), P.TAIL_HOURS: ("h", 1), P.UPDATE_MINUTES: ("min", 1),
    P.HEAT_LOSS: ("W/K", "any"), P.GROUND_LOSS: ("W/K", "any"), P.GROUND_TEMP: ("°C", "any"), P.AIR_CAPACITY: ("kWh/K", "any"),
    P.MASS_CAPACITY: ("kWh/K", "any"), P.COUPLING: ("W/K", "any"), P.GAIN_FRACTION: (None, "any"), P.GAIN_FIXED: ("kWh/d", "any"),
    P.SOLAR_GAIN: (None, "any"), P.SOLAR_TO_MASS: (None, "any"),
    P.HP_MAX_HEAT: ("kW", "any"), P.HP_KNEE_TEMP: ("°C", "any"), P.HP_HEAT_SLOPE: ("kW/K", "any"), P.HP_MIN_HEAT: ("kW", "any"),
    P.COP_BASE: (None, "any"), P.COP_TEMP: ("1/K", "any"), P.COP_LOAD: ("1/kW", "any"), P.COP_MIN: (None, "any"), P.COP_MAX: (None, "any"),
    P.START_RAMP: (None, "any"),
    P.NET_METERING_FEE: ("/kWh", "any"), P.EXPORT_FIXED_COSTS: ("/kWh", "any"), P.VAT_PERCENT: ("%", "any"), P.EXPORT_FEE: ("/kWh", "any"),
    P.PEAK_SHARE: ("%", 1),
    P.BAT_CAPACITY: ("kWh", "any"), P.BAT_CHARGE_KW: ("kW", "any"), P.BAT_DISCHARGE_KW: ("kW", "any"), P.BAT_CHARGE_EFF: (None, "any"),
    P.BAT_DISCHARGE_EFF: (None, "any"), P.BAT_WEAR: ("/kWh", "any"), P.BAT_MIN_SOC: ("%", 1), P.BAT_MAX_SOC: ("%", 1),
    P.DHW_POWER: ("kW", "any"), P.DHW_CHARGE_HOURS: ("h", 0.25), P.DHW_COP_REF: (None, "any"), P.DHW_COP_SLOPE: ("1/K", "any"),
    P.DHW_TARGET: ("°C", 0.5), P.DHW_LOW: ("°C", 0.5), P.DHW_HYST: ("K", 0.5), P.DHW_PER_DAY: (None, 1), P.DHW_MIN_GAP: ("h", 0.25),
    P.LEGIONELLA_HOURS: ("h", 0.25),
}

# config flow steps and options menu sections: step -> ordered keys
SECTIONS: dict[str, list[str]] = {
    "sources": [P.ROOM_TEMP_ENTITY, P.OUTDOOR_TEMP_ENTITY, P.WEATHER_ENTITY, P.FORECAST_BIAS, P.HEAT_OUTPUT_ENTITY,
                P.HEATING_ACTIVE_ENTITY],
    "energy": [P.IMPORT_ENERGY_ENTITY, P.EXPORT_ENERGY_ENTITY, P.PV_ENERGY_ENTITY, P.HP_ENERGY_ENTITY, P.PV_POWER_ENTITY,
               P.PV_FORECAST_ENTITIES],
    "prices": [P.PRICE_SOURCE, P.PRICE_ENTITY, P.PRICE_BREAKDOWN_ENTITY, P.AVOID_PEAKS, P.PEAK_ENTITY, P.PEAK_SHARE,
               P.NET_METERING, P.NET_METERING_ENTITY, P.NET_METERING_FEE, P.EXPORT_PRICE_ENTITY, P.EXPORT_FIXED_COSTS,
               P.VAT_PERCENT, P.EXPORT_FEE],
    "comfort": [P.MIN_TEMP, P.MAX_TEMP, P.HARD_MIN_TEMP, P.SAFETY_MARGIN, P.MIN_BLOCK_HOURS, P.START_COST, P.TAIL_HOURS,
                P.UPDATE_MINUTES],
    "building": [P.HEAT_LOSS, P.GROUND_LOSS, P.GROUND_TEMP, P.AIR_CAPACITY, P.MASS_CAPACITY, P.COUPLING, P.GAIN_FRACTION,
                 P.GAIN_FIXED, P.SOLAR_GAIN, P.SOLAR_TO_MASS],
    "heat_pump": [P.HP_MAX_HEAT, P.HP_KNEE_TEMP, P.HP_HEAT_SLOPE, P.HP_MIN_HEAT, P.COP_BASE, P.COP_TEMP, P.COP_LOAD, P.COP_MIN,
                  P.COP_MAX, P.START_RAMP],
    "battery": [P.PLAN_BATTERY, P.BATTERY_SOC_ENTITY, P.BAT_CAPACITY, P.BAT_CHARGE_KW, P.BAT_DISCHARGE_KW, P.BAT_CHARGE_EFF,
                P.BAT_DISCHARGE_EFF, P.BAT_WEAR, P.BAT_MIN_SOC, P.BAT_MAX_SOC, P.BAT_EXPORT],
    "hot_water": [P.DHW_MODE, P.DHW_SCHEDULE_ENTITY, P.DHW_TANK_ENTITY, P.LEGIONELLA_ENTITY, P.DHW_POWER, P.DHW_CHARGE_HOURS,
                  P.DHW_COP_REF, P.DHW_COP_SLOPE, P.DHW_TARGET, P.DHW_LOW, P.DHW_HYST, P.DHW_PER_DAY, P.DHW_MIN_GAP,
                  P.LEGIONELLA_HOURS],
}
# the initial config flow asks only for the essentials; everything else starts at its default
SETUP_STEPS = {
    "user": SECTIONS["sources"],
    "energy": SECTIONS["energy"],
    "prices": [P.PRICE_SOURCE, P.PRICE_ENTITY, P.PRICE_BREAKDOWN_ENTITY, P.PEAK_ENTITY, P.NET_METERING, P.NET_METERING_ENTITY,
               P.EXPORT_PRICE_ENTITY],
    "devices": [P.PLAN_BATTERY, P.BATTERY_SOC_ENTITY, P.DHW_MODE, P.DHW_SCHEDULE_ENTITY, P.DHW_TANK_ENTITY, P.LEGIONELLA_ENTITY],
}
SELECTS = {P.PRICE_SOURCE: list(P.PRICE_SOURCES), P.DHW_MODE: list(P.DHW_MODES)}
ALL_KEYS = set(P.ENTITY_KEYS) | {P.PV_FORECAST_ENTITIES} | set(P.NUMBERS) | set(P.BOOLEANS) | set(P.TEXTS)
