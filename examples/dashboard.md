# Dashboard examples

Example Lovelace cards that show the plans of Heat Planner. They read the three sensors of the integration
(`sensor.heat_planner_plan`, `sensor.heat_planner_battery`, `sensor.heat_planner_hot_water`) and its number
entities. Copy a card from [`cards/`](cards/) into a dashboard (edit dashboard → add card → Manual) and replace the
placeholder entities with your own. If you gave the integration another name, the entity ids start with that name
instead of `heat_planner`.

## Required custom cards

- [apexcharts-card](https://github.com/RomRider/apexcharts-card) (HACS) for the three charts.
- [browser_mod](https://github.com/thomasloven/hass-browser_mod) (HACS) only for the popup on the tile card.

The markdown, entities and tile cards are built in.

## Placeholder entities

Replace these with your own entities. Everything else in the cards comes from Heat Planner itself.

| Placeholder | Used in | What it must be |
|---|---|---|
| `sensor.room_temperature` | room temperature chart | the measured room temperature (°C), the same sensor you configured in Heat Planner |
| `sensor.heat_pump_heat_power` | room temperature chart | heat output of the heat pump in **kW** (heating and hot water together) |
| `binary_sensor.heat_pump_hot_water_active` | room temperature chart | `on` while the heat pump charges the hot water tank; a numeric sensor that is `0` outside hot water runs (e.g. a hot water flow temperature setpoint) also works. Used to split the measured heat into heating and hot water |
| `sensor.battery_soc` | battery chart | battery state of charge in % |
| `sensor.house_power` | battery chart | household consumption including the heat pump, in W (remove the `transform` if it is in kW) |
| `sensor.electricity_price` | price chart (optional series) | your all-in import price in EUR/kWh, for the recorded past; remove the series if you have none |
| `sensor.heat_planner_mass_temperature` | room temperature chart (optional) | the template sensor from [`template-sensors.yaml`](template-sensors.yaml); without it the mass line only shows the planned future |

## Cards

### Heat plan – [`cards/heat-plan-markdown.yaml`](cards/heat-plan-markdown.yaml)

A markdown card with the whole plan in words: the heating blocks (provisional ones beyond the known prices are
marked), the next block and the expected room temperature, the expected heat and electricity of the heat pump,
today's electricity (measured so far, the rest of the plan and the total, with an attribution to house, hot water,
heating, solar and battery) and a preview of tomorrow, the battery plan per period with prices, and the planned hot
water charges. Needs no other entities; set `beff` at the top to the one-way efficiency of your battery (only used
for the battery estimates).

### Room temperature and heating – [`cards/room-temperature-chart.yaml`](cards/room-temperature-chart.yaml)

Measured and predicted room temperature, the modelled temperature of the building mass, and measured and planned
heat output for space heating and hot water as areas. Shows 24 h back and 48 h ahead.

### Electricity prices – [`cards/price-chart.yaml`](cards/price-chart.yaml)

The import prices the planner uses (known prices solid, expected prices beyond them dashed) and the feed-in price
it computed, in ct/kWh. These series are published on `sensor.heat_planner_battery`, so battery planning must be
enabled.

### Battery – [`cards/battery-chart.yaml`](cards/battery-chart.yaml)

Measured and planned state of charge (planned forced charge or discharge from/to the grid highlighted in red) and
measured and expected household consumption. For a shorter view use `graph_span: 30h` and `offset: -6h`.

### Settings – [`cards/settings-markdown.yaml`](cards/settings-markdown.yaml)

A short summary of the active settings (comfort band, block length, battery and hot water mode, the fitted heat
loss and mass) with a link to the options dialog and to the history of the three sensors.

### Comfort – [`cards/comfort-entities.yaml`](cards/comfort-entities.yaml)

The four number entities, so the comfort band can be changed from the dashboard.

### Tile with popup – [`cards/status-tile.yaml`](cards/status-tile.yaml)

A tile with the plan state; a tap opens a browser_mod popup. Paste the cards above into its `cards:` list to have
the heat plan one tap away on another view.

## Optional template sensor

[`template-sensors.yaml`](template-sensors.yaml) records the estimated mass (floor) temperature from the
`state_estimate` attribute of the plan, which is not written to the recorder. Add it under `template:` in
`configuration.yaml` and restart (or reload template entities).

## Tips

- A heading card above the cards, e.g. `type: heading`, `heading: Heat Planner`, `icon: mdi:home-thermometer-outline`,
  keeps a sections view tidy.
- All charts use `grid_options: columns: full`, so in a sections view they take the whole width.
- Heat Planner only plans. If you control the heat pump, tank or battery with your own automations, you can add
  their switches and status helpers to the comfort card.
