"""Planner parameters: option keys, defaults and valid ranges.

This module is pure Python (no Home Assistant imports) so the engine can be used and tested offline.
The defaults describe an example house (well insulated, ~10 kW air/water heat pump, floor heating in a
heavy slab, 7 kWh battery). Fit or estimate the values for your own house; see the README.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------------------------- comfort / planner
MIN_TEMP = "min_temp"                  # degC, comfort floor for the room temperature
MAX_TEMP = "max_temp"                  # degC, comfort ceiling
HARD_MIN_TEMP = "hard_min_temp"        # degC, strongly penalised below this
MIN_BLOCK_HOURS = "min_block_hours"    # h, shortest heating block
START_COST = "start_cost"              # currency per compressor start (block)
SAFETY_MARGIN = "safety_margin"        # K added to the floor 24 h ahead (0.4x at 6 h, 0.6x at 12 h)
MARGIN_PENALTY = "margin_penalty"      # currency per K per 15-min slot inside the safety margin (x4 at the horizon end)
COMFORT_PENALTY = "comfort_penalty"    # currency per K per 15-min slot below min_temp or above max_temp (x10 below hard min)
TAIL_HOURS = "tail_hours"              # h planned beyond the last known price (expected prices)
UPDATE_MINUTES = "update_minutes"      # replan interval

# ----------------------------------------------------------------------------------------------- building (thermal model)
HEAT_LOSS = "heat_loss_w_k"            # W/K, room node -> outdoor (transmission + ventilation)
GROUND_LOSS = "ground_loss_w_k"        # W/K, mass node -> ground
GROUND_TEMP = "ground_temp"            # degC
AIR_CAPACITY = "air_capacity_kwh_k"    # kWh/K, room air + light furniture
MASS_CAPACITY = "mass_capacity_kwh_k"  # kWh/K, building mass (slab, floors, walls); heat pump heat goes here
COUPLING = "coupling_w_k"              # W/K, room node <-> mass node
GAIN_FRACTION = "internal_gain_fraction"   # share of household electricity (excl. heat pump) that ends up as heat
GAIN_FIXED = "internal_gain_kwh_day"   # kWh/day of other internal gains (people, ...)
SOLAR_GAIN = "solar_gain_factor"       # kWh heat into the house per kWh of PV production (PV used as irradiance proxy)
SOLAR_TO_MASS = "solar_to_mass"        # share of the solar gain that goes into the mass node
FORECAST_BIAS = "forecast_bias"        # 24 values (comma separated), weather forecast minus outdoor sensor per hour of day

# ----------------------------------------------------------------------------------------------- heat pump
HP_MAX_HEAT = "hp_max_heat_kw"         # kW heat output at full load at/below HP_KNEE_TEMP
HP_KNEE_TEMP = "hp_knee_temp"          # degC
HP_HEAT_SLOPE = "hp_heat_slope"        # kW less heat output per K above the knee
HP_MIN_HEAT = "hp_min_heat_kw"         # kW, lower limit of the full-load heat output
COP_BASE = "cop_base"                  # COP = base + temp*To - load*Q (Q = heat output kW), clipped
COP_TEMP = "cop_temp"
COP_LOAD = "cop_load"
COP_MIN = "cop_min"
COP_MAX = "cop_max"
START_RAMP = "start_ramp"              # fraction of the heat output delivered in the first 15 min of a block

# ----------------------------------------------------------------------------------------------- prices / export
NET_METERING = "net_metering"          # export is netted against import (export price = import price - fee)
NET_METERING_FEE = "net_metering_fee"  # currency/kWh below the import price when net metering
EXPORT_FIXED_COSTS = "export_fixed_costs"  # currency/kWh: all-in import price minus spot price incl. VAT (taxes, fees)
VAT_PERCENT = "vat_percent"
EXPORT_FEE = "export_fee"              # currency/kWh charged for export (without net metering)
AVOID_PEAKS = "avoid_peaks"            # no heating / hot water during peak price periods
PEAK_SHARE = "peak_share"              # % most expensive quarters per day treated as peak where no peak periods are known

# ----------------------------------------------------------------------------------------------- battery
PLAN_BATTERY = "plan_battery"
BAT_CAPACITY = "battery_capacity_kwh"
BAT_CHARGE_KW = "battery_charge_kw"
BAT_DISCHARGE_KW = "battery_discharge_kw"
BAT_CHARGE_EFF = "battery_charge_eff"
BAT_DISCHARGE_EFF = "battery_discharge_eff"
BAT_WEAR = "battery_wear"              # currency per kWh discharged
BAT_MIN_SOC = "battery_min_soc"        # %
BAT_MAX_SOC = "battery_max_soc"        # %
BAT_EXPORT = "battery_export"          # battery may discharge to the grid
# optional live sources for capacity and power limits (the numbers above are the fallback)
BAT_CAPACITY_ENTITY = "battery_capacity_entity"                # state = capacity (kWh or Wh)
BAT_CHARGE_POWER_ENTITY = "battery_charge_power_entity"        # state or attribute = max charge power (W or kW)
BAT_CHARGE_POWER_ATTR = "battery_charge_power_attribute"       # optional attribute of that entity
BAT_DISCHARGE_POWER_ENTITY = "battery_discharge_power_entity"  # state or attribute = max discharge power (W or kW)
BAT_DISCHARGE_POWER_ATTR = "battery_discharge_power_attribute" # optional attribute of that entity
# optional live limits for the current quarter only (e.g. the BMS's momentary allowed power, which drops to 0 at an empty
# or full battery): slot 0 is clipped to min(planning limit, live value), later slots use the planning limit
BAT_LIVE_CHARGE_POWER_ENTITY = "battery_live_charge_power_entity"
BAT_LIVE_CHARGE_POWER_ATTR = "battery_live_charge_power_attribute"
BAT_LIVE_DISCHARGE_POWER_ENTITY = "battery_live_discharge_power_entity"
BAT_LIVE_DISCHARGE_POWER_ATTR = "battery_live_discharge_power_attribute"
BAT_MIN_SOC_ENTITIES = "battery_min_soc_entities"              # list: minimum charge = highest available value (% or kWh)
BAT_MAX_SOC_ENTITY = "battery_max_soc_entity"                  # maximum charge (% or kWh)

# ----------------------------------------------------------------------------------------------- domestic hot water
DHW_MODE = "dhw_mode"                  # off | schedule (external schedule entity) | plan (the integration plans it)
DHW_POWER = "dhw_power_kw"             # kW electric while charging the tank (at DHW_COP_REF)
DHW_CHARGE_HOURS = "dhw_charge_hours"  # h the heat pump is busy per charge
DHW_COP_REF = "dhw_cop_ref"            # COP of a tank charge at 7 degC outdoor
DHW_COP_SLOPE = "dhw_cop_slope"        # COP change per K outdoor temperature
DHW_TARGET = "dhw_target_temp"         # degC reached by a planned charge
DHW_LOW = "dhw_low_temp"               # degC the heat pump keeps by itself (charges below low - hysteresis)
DHW_HYST = "dhw_hysteresis"            # K
DHW_PER_DAY = "dhw_charges_per_day"
DHW_MIN_GAP = "dhw_min_gap_hours"      # h between the end of one charge and the start of the next
LEGIONELLA_HOURS = "legionella_hours"  # h the heat pump is busy after the legionella entity turned on

DHW_MODES = ("off", "schedule", "plan")

# ----------------------------------------------------------------------------------------------- source entities
ROOM_TEMP_ENTITY = "room_temp_entity"            # room temperature sensor (the model's measurement)
OUTDOOR_TEMP_ENTITY = "outdoor_temp_entity"      # outdoor temperature sensor (history)
WEATHER_ENTITY = "weather_entity"                # hourly forecast (weather.get_forecasts)
HEAT_OUTPUT_ENTITY = "heat_output_entity"        # heat pump heat output power (W or kW), history
HEATING_ACTIVE_ENTITY = "heating_active_entity"  # optional: heat output counts as space heating only where this is > 0
PV_POWER_ENTITY = "pv_power_entity"              # PV power, history
PV_FORECAST_ENTITIES = "pv_forecast_entities"    # Solcast-style forecast sensors (detailedForecast attribute)
IMPORT_ENERGY_ENTITY = "import_energy_entity"    # grid import energy counter
EXPORT_ENERGY_ENTITY = "export_energy_entity"    # grid export energy counter
PV_ENERGY_ENTITY = "pv_energy_entity"            # PV production energy counter
HP_ENERGY_ENTITY = "heat_pump_energy_entity"     # heat pump electricity counter (subtracted from the household load)
BAT_CHARGE_ENERGY_ENTITY = "battery_charge_energy_entity"        # optional: battery charge energy counter (AC side)
BAT_DISCHARGE_ENERGY_ENTITY = "battery_discharge_energy_entity"  # optional: battery discharge energy counter (AC side)
PRICE_SOURCE = "price_source"                    # tibber | attributes
PRICE_ENTITY = "price_entity"                    # current price sensor (statistics for expected prices; attributes source)
PRICE_BREAKDOWN_ENTITY = "price_breakdown_entity"  # optional: all-in price with an `energy_price` attribute (live taxes)
PEAK_ENTITY = "peak_entity"                      # optional: entity with a `periods` attribute [{start, end}]
PRICE_FORECAST_ENTITY = "price_forecast_entity"  # optional: spot price series + forecast (raw_today/raw_tomorrow/raw_forecast)
NET_METERING_ENTITY = "net_metering_entity"      # optional: on/off entity overriding the net metering option
EXPORT_PRICE_ENTITY = "export_price_entity"      # optional: export price with net metering (live net metering fee)
BATTERY_SOC_ENTITY = "battery_soc_entity"        # battery state of charge (kWh or %)
DHW_SCHEDULE_ENTITY = "dhw_schedule_entity"      # text entity 'ISO/hours;ISO/hours' (dhw_mode schedule)
DHW_TANK_ENTITY = "dhw_tank_temp_entity"         # tank temperature
LEGIONELLA_ENTITY = "legionella_entity"          # on while a legionella cycle runs (heat pump busy)

PRICE_SOURCES = ("tibber", "attributes")
ENTITY_KEYS = (ROOM_TEMP_ENTITY, OUTDOOR_TEMP_ENTITY, WEATHER_ENTITY, HEAT_OUTPUT_ENTITY, HEATING_ACTIVE_ENTITY, PV_POWER_ENTITY,
               IMPORT_ENERGY_ENTITY, EXPORT_ENERGY_ENTITY, PV_ENERGY_ENTITY, HP_ENERGY_ENTITY, PRICE_ENTITY,
               PRICE_BREAKDOWN_ENTITY, PEAK_ENTITY, PRICE_FORECAST_ENTITY, NET_METERING_ENTITY, EXPORT_PRICE_ENTITY, BATTERY_SOC_ENTITY,
               DHW_SCHEDULE_ENTITY, DHW_TANK_ENTITY, LEGIONELLA_ENTITY, BAT_CAPACITY_ENTITY, BAT_CHARGE_POWER_ENTITY,
               BAT_DISCHARGE_POWER_ENTITY, BAT_MAX_SOC_ENTITY, BAT_CHARGE_ENERGY_ENTITY, BAT_DISCHARGE_ENERGY_ENTITY,
               BAT_LIVE_CHARGE_POWER_ENTITY, BAT_LIVE_DISCHARGE_POWER_ENTITY)
# energy counters of the household balance: role -> option key (household = import - export + PV - heat pump - battery net)
ENERGY_ROLES = {"import": IMPORT_ENERGY_ENTITY, "export": EXPORT_ENERGY_ENTITY, "pv": PV_ENERGY_ENTITY, "heat_pump": HP_ENERGY_ENTITY,
                "battery_charge": BAT_CHARGE_ENERGY_ENTITY, "battery_discharge": BAT_DISCHARGE_ENERGY_ENTITY}
# options holding a list of entities
ENTITY_LIST_KEYS = (PV_FORECAST_ENTITIES, BAT_MIN_SOC_ENTITIES)
# optional free-text options (absent = not used)
OPTIONAL_TEXTS = (BAT_CHARGE_POWER_ATTR, BAT_DISCHARGE_POWER_ATTR, BAT_LIVE_CHARGE_POWER_ATTR, BAT_LIVE_DISCHARGE_POWER_ATTR)

# key: (default, min, max)
NUMBERS: dict[str, tuple[float, float, float]] = {
    MIN_TEMP: (18.5, 10.0, 25.0),
    MAX_TEMP: (22.0, 12.0, 28.0),
    HARD_MIN_TEMP: (18.0, 8.0, 25.0),
    MIN_BLOCK_HOURS: (2.5, 0.25, 8.0),
    START_COST: (0.10, 0.0, 5.0),
    SAFETY_MARGIN: (0.5, 0.0, 3.0),
    MARGIN_PENALTY: (2.0, 0.0, 100.0),
    COMFORT_PENALTY: (20.0, 0.0, 100.0),
    TAIL_HOURS: (24, 0, 48),
    UPDATE_MINUTES: (15, 5, 120),
    HEAT_LOSS: (114.83745496075469, 1.0, 2000.0),
    GROUND_LOSS: (20.0, 0.0, 1000.0),
    GROUND_TEMP: (10.0, -10.0, 25.0),
    AIR_CAPACITY: (1.9944733527161391, 0.05, 50.0),
    MASS_CAPACITY: (19.603056580237276, 0.1, 500.0),
    COUPLING: (2589.196132745592, 10.0, 50000.0),
    GAIN_FRACTION: (0.85, 0.0, 1.0),
    GAIN_FIXED: (3.0, 0.0, 50.0),
    SOLAR_GAIN: (0.23108883242261152, 0.0, 3.0),
    SOLAR_TO_MASS: (0.0, 0.0, 1.0),
    HP_MAX_HEAT: (10.0, 0.5, 50.0),
    HP_KNEE_TEMP: (5.0, -20.0, 20.0),
    HP_HEAT_SLOPE: (0.6, 0.0, 5.0),
    HP_MIN_HEAT: (6.0, 0.1, 50.0),
    COP_BASE: (6.0, 0.5, 10.0),
    COP_TEMP: (0.14, -1.0, 1.0),
    COP_LOAD: (0.29, -1.0, 2.0),
    COP_MIN: (2.5, 0.5, 10.0),
    COP_MAX: (6.5, 0.5, 15.0),
    START_RAMP: (0.6, 0.0, 1.0),
    NET_METERING_FEE: (0.0248, -1.0, 1.0),
    EXPORT_FIXED_COSTS: (0.1357, -1.0, 2.0),
    VAT_PERCENT: (21.0, 0.0, 50.0),
    EXPORT_FEE: (0.0, -1.0, 1.0),
    PEAK_SHARE: (20.0, 0.0, 60.0),
    BAT_CAPACITY: (7.0, 0.2, 100.0),
    BAT_CHARGE_KW: (3.5, 0.0, 50.0),
    BAT_DISCHARGE_KW: (3.5, 0.0, 50.0),
    BAT_CHARGE_EFF: (0.922, 0.5, 1.0),
    BAT_DISCHARGE_EFF: (0.922, 0.5, 1.0),
    BAT_WEAR: (0.035, 0.0, 1.0),
    BAT_MIN_SOC: (0.0, 0.0, 100.0),
    BAT_MAX_SOC: (100.0, 0.0, 100.0),
    DHW_POWER: (2.0, 0.1, 20.0),
    DHW_CHARGE_HOURS: (0.75, 0.25, 6.0),
    DHW_COP_REF: (4.1, 1.0, 8.0),
    DHW_COP_SLOPE: (0.06, -0.5, 0.5),
    DHW_TARGET: (50.0, 30.0, 70.0),
    DHW_LOW: (39.0, 20.0, 65.0),
    DHW_HYST: (4.0, 0.0, 20.0),
    DHW_PER_DAY: (1, 0, 4),
    DHW_MIN_GAP: (8.0, 0.0, 24.0),
    LEGIONELLA_HOURS: (1.5, 0.0, 12.0),
}
BOOLEANS: dict[str, bool] = {
    NET_METERING: False,
    AVOID_PEAKS: True,
    PLAN_BATTERY: False,
    BAT_EXPORT: True,
}
TEXTS: dict[str, str] = {
    FORECAST_BIAS: "0",
    DHW_MODE: "off",
    PRICE_SOURCE: "tibber",
}

# internal model constants (not exposed as options)
DISTURBANCE_HOURS = 12.0           # decay time of the estimated unmeasured-gains state in the prediction
KALMAN_Q_HOURLY = (0.0004, 0.0001, 0.0025)   # process noise per hour for (room, mass, disturbance)
KALMAN_R = 0.0009                  # measurement noise of the room sensor (K^2)


def defaults() -> dict:
    out = {k: v[0] for k, v in NUMBERS.items()}
    out.update(BOOLEANS)
    out.update(TEXTS)
    return out


def parse_bias(text) -> list[float]:
    """'0' or 24 comma/space separated values -> list of 24 floats (one value is used for every hour)"""
    if isinstance(text, (list, tuple)):
        vals = [float(v) for v in text]
    else:
        vals = [float(v) for v in str(text or "0").replace(";", ",").replace(" ", ",").split(",") if v.strip()]
    if len(vals) == 1:
        vals = vals * 24
    if len(vals) != 24:
        raise ValueError("forecast bias needs 1 or 24 values")
    return vals


def normalise(opts: dict) -> dict:
    """defaults + options; numbers clipped to their range; consistent comfort band"""
    p = defaults()
    p.update({k: v for k, v in (opts or {}).items() if v is not None})
    for k, (d, lo, hi) in NUMBERS.items():
        try:
            v = float(p[k])
        except (TypeError, ValueError):
            v = d
        p[k] = min(max(v, lo), hi)
    for k, d in BOOLEANS.items():
        p[k] = bool(p[k])
    if p[DHW_MODE] not in DHW_MODES:
        p[DHW_MODE] = "off"
    if p[PRICE_SOURCE] not in PRICE_SOURCES:
        p[PRICE_SOURCE] = "tibber"
    p[HARD_MIN_TEMP] = min(p[HARD_MIN_TEMP], p[MIN_TEMP])
    if p[MAX_TEMP] < p[MIN_TEMP] + 0.5:
        p[MAX_TEMP] = p[MIN_TEMP] + 0.5
    if p[BAT_MAX_SOC] < p[BAT_MIN_SOC]:
        p[BAT_MAX_SOC] = p[BAT_MIN_SOC]
    p[HP_MIN_HEAT] = min(p[HP_MIN_HEAT], p[HP_MAX_HEAT])
    p[COP_MAX] = max(p[COP_MAX], p[COP_MIN])
    p["_bias"] = parse_bias(p[FORECAST_BIAS])
    return p
