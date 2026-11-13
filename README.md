# Heat Planner

A predictive planner for Home Assistant that works out **when** a heat pump should heat the house, when it
should charge the hot water tank, and what a home battery should do, so that the electricity bill is as low as
possible while the room stays within your comfort band.

**Heat Planner is read-only.** It computes plans and publishes them as sensors. It never switches, sets or
otherwise controls a heat pump, boiler, thermostat or battery, and it never calls a service that changes
anything. If you want to act on the plan, write your own automations that read the sensors.

It is built for dynamic (hourly or quarter-hourly) electricity prices, a well-insulated house with a
heavy floor (where heat can be stored for many hours) and, optionally, solar panels and a home battery.

## How it works

Every 15 minutes (configurable) the planner:

1. **Estimates the current thermal state** of the house from the last two days of history (room temperature,
   outdoor temperature, heat pump heat output, PV power, household electricity) with a Kalman filter. Only the
   room temperature is measured; the temperature of the building mass and an "unmeasured gains" term are
   estimated.
2. **Builds forecasts** on a 15-minute grid from now until the last known price plus a tail (default 24 h):
   outdoor temperature from the weather forecast, PV from the PV forecast, household electricity from the median
   of the same hour over the last 7 days, prices from the price source (beyond the known prices: a price forecast entity if configured, otherwise
   the median of the same hour over the last 7 days; marked *provisional*).
3. **Plans hot water** (optional): the tank charges are placed at the cheapest moments, at least the configured
   gap apart, outside peak periods and outside a running legionella cycle.
4. **Plans space heating and the battery jointly** with dynamic programming over 15-minute slots: heating
   blocks of at least the minimum length at the heat pump's full-load output, battery charge/discharge in
   0.2 kWh steps. The heat pump cannot heat while it charges the tank, so the hot water charges are busy slots.
5. **Publishes the result** in three sensors.

### Thermal model

A two-node RC model with a disturbance state:

```
Cr dTr/dt = Hrs (Ts - Tr) - H (Tr - To) + qi + kl E + g (1 - fs) PV + d
Cs dTs/dt = Hrs (Tr - Ts) - Hg (Ts - Tg) + Q + g fs PV
dd/dt     = -d / 12 h
```

| Symbol | Option | Meaning |
|---|---|---|
| Tr, Ts | | room air and building mass temperature |
| H | Heat loss (W/K) | room to outdoor (transmission + ventilation) |
| Hg, Tg | Ground coupling (W/K), ground temperature | mass to ground |
| Cr, Cs | Air / mass heat capacity (kWh/K) | |
| Hrs | Air-mass coupling (W/K) | |
| kl, qi | Internal gains: share of household electricity, other gains (kWh/day) | |
| g, fs | Solar gain factor (kWh heat per kWh PV), share into the mass | PV production is used as irradiance proxy |
| Q | | heat pump heat output, goes into the mass node (floor heating) |
| d | | unmeasured gains, estimated by the Kalman filter |

Heat pump: full-load heat output `Qmax` at and below the knee temperature, dropping by `slope` kW per K above it
(never below the minimum). COP = `base + temp * To - load * Q`, clipped to `[min, max]`. The first 15 minutes of a
block deliver a configurable share of the output (start-up).

The defaults describe an example house (about 115 W/K heat loss + 20 W/K to the ground, 2 kWh/K air, 20 kWh/K mass,
a 10 kW air-to-water heat pump). **Fit the parameters for your own house.** A simple approach: export a few weeks
of hourly room temperature, outdoor temperature, heat output, PV and household electricity from the recorder and
fit the model above with multi-step prediction error (e.g. 6 to 36 h ahead). The heat loss and the internal gains
are correlated; fix the internal gains from your electricity data and fit the rest.

### Optimisation

- Objective: import cost minus export revenue, plus battery wear, plus a cost per heating start, plus comfort
  penalties (below the minimum, above the maximum, much higher below the hard minimum), minus a terminal value
  for heat stored in the house and energy stored in the battery at the end of the horizon.
- Comfort penalties are soft costs per K per 15-minute slot: `comfort_penalty` (default 20) below the minimum or
  above the maximum (10x below the hard minimum) and `margin_penalty` (default 2) inside the safety margin
  (4x at the end of the horizon). A higher margin penalty makes the planner heat earlier and safer, a lower one
  lets it save more by dipping into the margin.
- Safety margin: the minimum is raised far ahead (full margin from 24 h, 60 % at 12 h, 40 % at 6 h) because
  forecasts and the model get less certain.
- Peak periods: no heating block may start or run inside a peak period (a running block must have reached the
  minimum length).
- Running blocks: once a planned heating block has started, it stays fixed until its planned end (the planner may
  still extend it); only the time after that end is planned freely. It is reported from its original start and in
  `running_block`.
- Post-check: the optimiser keeps temperatures on a grid (0.1 K room, 0.2 K mass-room) and interpolates between
  grid points, which can misjudge heat that is stored many hours before it is needed (for example a short block at
  an expensive moment instead of a cheaper one earlier in the night). After the optimiser, a local search moves
  every heating block earlier or later (up to 4 h), makes it longer or shorter (up to 2 h), drops or merges blocks,
  and scores each candidate with the exact model and the same objective. The result only replaces the optimiser's
  schedule when it is cheaper. With battery planning the battery schedule is kept as optimised. The attribute
  `post_check` shows the objective before and after, the number of accepted moves and the runtime.
- Runtime on a small machine: heating only well under a second; heating + battery a few seconds (run in the
  executor, not in the event loop).

### Hot water

Modes:

- `off`: hot water is ignored.
- `schedule`: an external text entity with `ISO-start/hours;ISO-start/hours` blocks (for example written by
  another planner); the first *charge duration* of every block counts as busy at the charge power.
- `plan`: the planner places *charges per day* tank charges per calendar day at the lowest marginal electricity
  cost (import price, or missed export revenue when the sun covers it), *minimum time between charges* apart,
  outside peak periods and legionella cycles. The electric power of a charge scales with the hot water COP
  (lower when it is cold outside). If the tank sensor reached *target - hysteresis* today, today's charge counts
  as done. If the tank is below *low - hysteresis* the heat pump is expected to charge by itself right away.

Hot water is planned first; heating and battery are then optimised around it (sequential, which keeps the
runtime low; space heating is flexible enough that the loss is small).

## Inputs

| Input | Required | Notes |
|---|---|---|
| Room temperature sensor | yes | with long-term statistics (state_class measurement) |
| Outdoor temperature sensor | yes | the sensor the model was fitted on |
| Weather entity | yes | hourly forecast via `weather.get_forecasts`; an hourly bias profile (forecast minus sensor) can be set |
| Heat pump heat output (W/kW) | recommended | history for the state estimate; optional "space heating active" sensor separates hot water |
| Grid import energy | yes | household load profile = import - export + PV - heat pump |
| Grid export, PV energy, heat pump energy | optional | |
| PV power | optional | history for the solar gains |
| PV forecast | optional | sensors with a `detailedForecast` attribute (Solcast) |
| Prices | yes | `tibber` (service `tibber.get_prices`) or `attributes` (a sensor with a price list in its attributes, e.g. Nord Pool `raw_today`/`raw_tomorrow`, ENTSO-e `prices`) |
| Price sensor statistics | yes | expected prices beyond the known ones |
| Price breakdown sensor | optional | all-in price with an `energy_price` attribute: taxes per kWh live |
| Price forecast entity | optional | spot price series with `raw_today` / `raw_tomorrow` / `raw_forecast` attributes (e.g. the price forecast series of [Energy Price Forecast EU](https://github.com/BackupBaTTerY/energypriceforecast-home-assistant), base price). Used for slots without a known price as spot × (1 + VAT) + fixed part; the fixed part is the median difference over the slots where both prices are known (at least 8), otherwise the fixed costs setting / price breakdown sensor |
| Peak periods entity | optional | `periods` attribute `[{start, end}]` (e.g. from a price integration); otherwise the most expensive share of each day |
| Net metering switch, export price sensor | optional | export price = import price - fee with net metering; otherwise (price - taxes) / (1 + VAT) - export fee |
| Battery state of charge | for battery planning | kWh or % |
| Battery capacity, max charge / discharge power entities | optional | read live at every replan from the state or a named attribute (e.g. `max_charge_power` / `max_discharge_power` of a power control entity); W/kW and Wh/kWh by unit; the capacity and power numbers are the fallback when unavailable. The battery sensor shows the values used (`capacity`, `charge_kw`, `discharge_kw`, `min_soc`, `max_soc`, `live_limits`) |
| Battery minimum / maximum charge entities | optional | % or kWh, read live; minimum = highest available of the listed entities (e.g. discharge limit and backup reserve); the minimum / maximum charge numbers are the fallback |
| Tank temperature, legionella entity, external schedule | optional | hot water |

Units are taken from the entities (`W`/`kW`, `Wh`/`kWh`, `ct/kWh`/`EUR/kWh`, `°C`/`°F`).

## Outputs

`sensor.<name>_plan`: state = heating blocks within the known prices as `ISO-start/hours;...` (or `none`).
Attributes include `blocks`, `blocks_tail` (provisional, beyond the known prices), `next_block_start`,
`tr` / `ts` / `tr_tail` (predicted room / mass temperature per quarter), `tr_min`, `tr_max`, `tr_next_block`,
`tr_min_until_next_block`, `heat_kwh`, `elec_kwh`, `cost`, `energy` (house, heat pump, hot water, solar, import,
export, average prices), `cost_breakdown` (house, heat pump, solar, battery, wear, total), `blocks_without_battery`,
`net_metering`, `net_metering_fee`, `price_forecast` (entity, slots used, learned fixed part), `running_block`
(start and end of a block that already started, else null), `today`, `horizon_end`, `created`,
`runtime_s`, `error`.

`today` is the money of the current day in three parts: `realised` (local midnight until the current quarter, from the
5-minute statistics of the import, export, PV, heat pump and battery entities and today's prices), `rest` (the plan
from now until midnight) and `total` (the sum), plus `tomorrow` (plan for the next day, provisional;
`tomorrow_known_share` = share of quarters with known prices). Each part has the real meter flows (`import_kwh`,
`import_cost`, `export_kwh`, `export_revenue`, `net`, average prices) and an attribution that adds up to `net`:
`house` (without hot water), `boiler` (hot water), `heating`, `solar_self` (saving from self-use), `solar_export`
(export revenue of the surplus), `battery`, `total`, with the matching kWh and `battery_wear` (not in the total).
Positive = cost, negative = revenue or saving. In `realised` the house is the remainder behind the meter, hot water is
the heat pump electricity while space heating is off (needs the heating-active entity), and the battery energy follows
from its level; meter timing differences end up in `battery`.

`sensor.<name>_battery`: state = readable summary (e.g. `Charge 12:00-13:00 3.5 kWh (solar)`), or `off`.
Attributes: `blocks` (action, start, end, kWh, via), `soc` (kWh per quarter), `code` (`start|letters`, one letter per
quarter: `C` charge from solar, `F` charge from the grid, `D` discharge for the house, `X` discharge with export,
`P` idle), `prices`, `feed_in_prices`.

`sensor.<name>_hot_water`: state = readable summary of the planned tank charges. Attributes: `mode`, `blocks`
(start, end, kWh, price, provisional), `schedule` (`ISO-start/hours;...`), `next_start`, `done_today`,
`tank_temp`, `legionella_active`, `kwh`.

The summaries follow the Home Assistant language (English or Dutch). The previous plan is kept when a run
fails; the error is shown in the `error` attribute.

Service `heat_planner.replan` computes a new plan right away.

Number entities `number.<name>_min_temp`, `_max_temp`, `_hard_min_temp` and `_safety_margin` (configuration) show and
change the same comfort options as the options dialog, so they can be put on a dashboard. Changing them re-runs the
planner without reloading the integration; values that break the rules of the options dialog are rejected.

## Installation

1. HACS → Integrations → menu → Custom repositories → add `https://github.com/TAW-V/heat-planner` (type
   Integration), install **Heat Planner** and restart Home Assistant.
   Manual: copy `custom_components/heat_planner` into your `config/custom_components` folder and restart.
2. Settings → Devices & services → Add integration → **Heat Planner**, pick your sensors.
3. Tune the model and comfort settings under **Configure** (groups: sources, energy, prices, comfort, building,
   heat pump, battery, hot water).

No extra Python packages are needed (only numpy, which ships with Home Assistant).

Optional one-time YAML import (for scripted migrations): a `heat_planner:` block in `configuration.yaml` with the
option keys (see `custom_components/heat_planner/engine/params.py`) creates a config entry at startup; remove the
block afterwards.

## Limitations

- Planning only: nothing is controlled. Results depend on how well the model parameters fit your house.
- Heat pump modulation is not modelled: a heating block runs at full-load output. Flow temperature, heating curve
  and room-by-room effects are not modelled; one room sensor represents the house.
- The model is linear and fitted around normal operation; recheck it after the first frost period.
- Hot water is planned before heating (sequential), not fully jointly.
- PV beyond the forecast horizon uses the last forecast day at half weight; prices beyond the known prices use
  the price forecast entity where configured and available, otherwise last week's hourly medians.
- Only `°C`-based comfort settings; temperature sensors in `°F` are converted.
- A running heating block is known from the previous planning run only; right after a restart of Home Assistant
  a block that already started is not kept by the planner.

## Development

`custom_components/heat_planner/engine` is pure Python + numpy and has no Home Assistant imports; run the offline
tests with `python -m pytest tests`.

## License

MIT
