# Connection Italy to its neighbors (France, Swiss and Austria)
# %% package and data import
import pandas as pd
import matplotlib.pyplot as plt
import pypsa
import networkx as nx

data_solar = pd.read_csv('data/pv_optimal.csv',sep=';')
data_solar.index = pd.DatetimeIndex(data_solar['utc_time'])

data_wind = pd.read_csv('data/onshore_wind_1979-2017.csv',sep=';')
data_wind.index = pd.DatetimeIndex(data_wind['utc_time'])

data_el = pd.read_csv('data/electricity_demand.csv',sep=';')
data_el.index = pd.DatetimeIndex(data_el['utc_time'])

data_solar.head()

# %% 
# Costs calculation
year_economic_data = 2020

url = f"https://raw.githubusercontent.com/PyPSA/technology-data/v0.11.0/outputs/costs_{year_economic_data}.csv"
costs = pd.read_csv(url, index_col=[0, 1])

costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
costs.unit = costs.unit.str.replace("/kW", "/MW")

defaults = {
    "FOM": 0,
    "VOM": 0,
    "efficiency": 1,
    "fuel": 0,
    "investment": 0,
    "lifetime": 25,
    "discount rate": 0.07,
}
costs = costs.value.unstack().fillna(defaults)

costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]

def annuity(r, n):
    return r / (1.0 - 1.0 / (1.0 + r) ** n)

techs = ["onwind", "solar", "OCGT", "nuclear"]
for tech in techs:
    costs.at[tech, "marginal_cost"] = costs.at[tech,"VOM"] + costs.at[tech, "fuel"] / costs.at[tech, "efficiency"]
    annuity_applied = costs.apply(lambda x: annuity(x["discount rate"], x["lifetime"]), axis=1)
    costs.at[tech, "capital_cost"] = (annuity_applied[tech] + costs.at[tech, "FOM"] / 100) * costs.at[tech, "investment"]
    print(f"{tech}: Capital Cost = {costs.at[tech, 'capital_cost']} EUR/MW/a, Marginal Cost = {costs.at[tech, 'marginal_cost']} EUR/MWh")
techs.append("battery storage")

# Battery costs — lithium-ion typical values (2020, PyPSA technology-data v0.11)
# Inverter:        24,678 €/MW/a    (power electronics, per MW)
# Energy storage:  12,894 €/MWh/a  (cells, per MWh)
# With max_hours=2: total capital cost per MW of power capacity:
capital_cost_battery = (annuity(20, 0.07) * 310_000 * (1 + 0.0)   # inverter: ~24k €/MW/a
                        + 2 * annuity(20, 0.07) * 150_000)         # storage:  ~12k €/MWh/a × 2 MWh/MW
# Alternative: use flat literature values directly
capital_cost_battery = 24_678 + 2 * 12_894   # €/MW/a  (total per MW of power)
eta_battery = 0.96   # one-way efficiency (round-trip = 0.96² ≈ 0.92)

print(f"Battery capital cost: {capital_cost_battery:,.0f} €/MW/a")
print(f"  = {24_678:,.0f} €/MW/a (inverter) + 2 × {12_894:,.0f} €/MWh/a (storage)")
print(f"Battery one-way efficiency: {eta_battery:.2f}  |  round-trip: {eta_battery**2:.3f}")

# %% 
# Network Creation
# Nodes and loads
# ----------------------------
year = 2010
n = pypsa.Network()
hours_in_year = pd.date_range(f'{year}-01-01 00:00Z',
                              f'{year}-12-31 23:00Z',
                              freq='h')
n.set_snapshots(hours_in_year.values)

n.add("Bus", "ITA", v_nom = 400, x=0, y=0)
neighboring_countries = ["FRA", "CHE", "AUT"]
n.add("Bus", "FRA", v_nom = 400, x=-1, y=1)
n.add("Bus", "CHE", v_nom = 400, x=0, y=1)
n.add("Bus", "AUT", v_nom = 400, x=1, y=1)
n.buses

n.add("Load", "demand_ITA", bus="ITA", p_set=data_el["ITA"].values)
for country in neighboring_countries:
    n.add("Load",
        f"demand_{country}",
        bus=country,
        p_set=data_el[country].values)
    
# Generators
# ----------------------------
carriers = techs  # all except battery storage
carrier_colors = ["dodgerblue", "gold", "indianred", "purple", "slategray"]
n.add(
    "Carrier",
    carriers,
    color=carrier_colors
)

# Italy has OCGT, onshore wind and solar
n.add(
    "Generator",
    "OCGT_ITA",
    bus="ITA",
    carrier="OCGT",
    capital_cost=costs.at["OCGT", "capital_cost"],
    marginal_cost=costs.at["OCGT", "marginal_cost"],
    efficiency=costs.at["OCGT", "efficiency"],
    p_nom_extendable=True,
)

CF_wind_ITA = data_wind["ITA"][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in n.snapshots]]
n.add(
    "Generator",
    "onwind_ITA",
    bus="ITA",
    carrier="onwind",
    p_max_pu=CF_wind_ITA.values,
    capital_cost=costs.at["onwind", "capital_cost"],
    marginal_cost=costs.at["onwind", "marginal_cost"],
    efficiency=costs.at["onwind", "efficiency"],
    p_nom_extendable=True,
)

CF_solar_ITA = data_solar["ITA"][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in n.snapshots]]
n.add(
    "Generator",
    "solar_ITA",
    bus="ITA",
    carrier="solar",
    p_max_pu= CF_solar_ITA.values,
    capital_cost=costs.at["solar", "capital_cost"],
    marginal_cost=costs.at["solar", "marginal_cost"],
    efficiency=costs.at["solar", "efficiency"],
    p_nom_extendable=True,
)

# France has Nuclear and Wind
n.add(
    "Generator",
    name="nuclear_FRA",
    bus="FRA",
    carrier="nuclear",
    marginal_cost=costs.at["nuclear", "marginal_cost"],
    capital_cost=costs.at["nuclear", "capital_cost"],
    p_nom_extendable=True,
)
CF_wind_FRA = data_wind["FRA"][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in n.snapshots]]
n.add(
    "Generator",
    "onwind_FRA",
    bus="FRA",
    carrier="onwind",
    p_max_pu=CF_wind_FRA.values,
    capital_cost=costs.at["onwind", "capital_cost"],
    marginal_cost=costs.at["onwind", "marginal_cost"],
    efficiency=costs.at["onwind", "efficiency"],
    p_nom_extendable=True,
)

# Austria has onshore wind and OCGT
CF_wind_AUT = data_wind["AUT"][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in n.snapshots]]
n.add(
    "Generator",
    "onwind_AUT",
    bus="AUT",
    carrier="onwind",
    p_max_pu=CF_wind_AUT.values,
        capital_cost=costs.at["onwind", "capital_cost"],
        marginal_cost=costs.at["onwind", "marginal_cost"],
    p_nom_extendable=True,
)
n.add(
    "Generator",
    "OCGT_AUT",
    bus="AUT",
    carrier="OCGT",
    capital_cost=costs.at["OCGT", "capital_cost"],
    marginal_cost=costs.at["OCGT", "marginal_cost"],
    efficiency=costs.at["OCGT", "efficiency"],
    p_nom_extendable=True,
)

# Switzerland has solar and OCGT
CF_solar_CHE = data_solar["CHE"][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in n.snapshots]]
n.add(
    "Generator",
    "solar_CHE",
    bus="CHE",
    carrier="solar",
    p_max_pu= CF_solar_CHE.values,
    capital_cost=costs.at["solar", "capital_cost"],
    marginal_cost=costs.at["solar", "marginal_cost"],
    p_nom_extendable=True,
)
n.add(
    "Generator",
    "OCGT_CHE",
    bus="CHE",
    carrier="OCGT",
    capital_cost=costs.at["OCGT", "capital_cost"],
    marginal_cost=costs.at["OCGT", "marginal_cost"],
    efficiency=costs.at["OCGT", "efficiency"],
    p_nom_extendable=True,
)

# Storage System for Italy
# ----------------------------
eta = costs.at["battery inverter", "efficiency"]   # one-way efficiency

n.add("StorageUnit", "battery_ITA", bus="ITA", carrier="battery storage",
      max_hours=2,
      capital_cost=capital_cost_battery,
      efficiency_store=eta,
      efficiency_dispatch=eta,
      p_nom_extendable=True,
      cyclic_state_of_charge=True)

# Lines
# ----------------------------
# For HV lines, the ratio x/r is usually around 10. 
# As x is imposed to 0.1, we choose r=0.01 for all lines.
# Italy - France
n.add("Line",
    "ITA-FRA",bus0="ITA",bus1="FRA",
    s_nom=3000,x=0.1,r=0.01,
)

# Italy - Switzerland
n.add("Line",
    "ITA-CHE",bus0="ITA",bus1="CHE",
    s_nom=4000,x=0.1,r=0.01,
)

# Italy - Austria
n.add("Line",
    "ITA-AUT",bus0="ITA",bus1="AUT",
    s_nom=1200,x=0.1,r=0.01,
)

# France - Switzerland
n.add("Line",
    "FRA-CHE",bus0="FRA",bus1="CHE",
    s_nom=3500,x=0.1,r=0.01,
)

# %%
# Create a directed graph
G = nx.DiGraph()
G.add_nodes_from(n.buses.index)
for i, line in n.lines.iterrows():
    G.add_edge(line.bus0, line.bus1)

pos = {
    'FRA': (-0.4, 0.4),
    'CHE': (0, 0.4),
    'AUT': (0.4, 0.4),
    'ITA': (0, 0)
}
# plt.figure(figsize=(8, 6))
# nx.draw(
#     G, pos,
#     with_labels=True, 
#     node_color='lightblue', 
#     node_size=2500, 
#     font_weight='bold',
#     edge_color='gray',
#     width=2,
#     arrows=True,
#     arrowstyle='-|>',
#     arrowsize=20
# )

# #plt.title("Italy and Neighbors", fontsize=14)
# plt.savefig('pics/network.png', dpi=150)
# plt.show()

# %% 
# Solve
n.optimize(store_basis=True, keep_references=True)
# Network.optimize.fix_optimal_capacities :
# when a capacity expansion optimization was 
# already performed and a operational optimization should be done afterwards.

# %% Dispatch plot
# Get the dispatch time series for all generators and the italian storage units
dispatch_df = n.generators_t.p
dispatch_df["battery_ITA"] = n.storage_units_t.p.clip(lower=0)  # only discharge counts as generation
carriers_series = n.generators.carrier
carriers_series["battery_ITA"] = "battery storage"
# Group the dispatch by carrier and sum them up
carrier_dispatch_df = dispatch_df.T.groupby(carriers_series).sum().T
# Now carrier_dispatch_df is a DataFrame where each column represents a carrier 
# (e.g., 'solar', 'onwind', 'OCGT', 'nuclear') and contains the total 
# power generated by that carrier at each timestep.
print(carrier_dispatch_df.head())

summer_start = pd.Timestamp(f'{year}-07-06')
summer_slice  = slice(summer_start, summer_start + pd.Timedelta(hours=167))

fig, ax = plt.subplots(figsize=(12, 4))
dem = pd.DataFrame(index=carrier_dispatch_df.index)
for country in ["ITA", "FRA", "CHE", "AUT"]:
    dem[country] = n.loads_t.p[f'demand_{country}']

conso_demand_ts = dem.sum(axis=1)
total_demand = conso_demand_ts - n.storage_units_t.p.clip(upper=0).sum(axis=1)  # add storage charging to total demand

ax.stackplot(carrier_dispatch_df.index,
    carrier_dispatch_df['onwind'],
    carrier_dispatch_df['solar'],
    carrier_dispatch_df['OCGT'],
    carrier_dispatch_df['nuclear'],
    carrier_dispatch_df['battery storage'],
    labels=techs,
    colors=carrier_colors + ['slategray'], alpha=0.85
)
ax.plot(conso_demand_ts.index, conso_demand_ts.values, color='black', lw=1.5, label='Consumption Demand')
ax.plot(total_demand.index, total_demand.values, color='red', lw=1.5, linestyle='--', label='Total Demand')
ax.set_title(f'Dispatch - Summer Week (July {year})')
ax.set_ylabel('Power [MW]')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_xlabel('Date')
ax.set_xlim(summer_start, summer_start + pd.Timedelta(hours=167))
fig.tight_layout()
plt.show()

winter_start = pd.Timestamp(f'{year}-01-06')
winter_slice  = slice(winter_start, winter_start + pd.Timedelta(hours=167))

fig, ax = plt.subplots(figsize=(12, 4))
dem = pd.DataFrame(index=carrier_dispatch_df.index)
for country in ["ITA", "FRA", "CHE", "AUT"]:
    dem[country] = n.loads_t.p[f'demand_{country}']

conso_demand_ts = dem.sum(axis=1)
total_demand_ts = conso_demand_ts - n.storage_units_t.p.clip(upper=0).sum(axis=1)  # add storage charging to total demand

ax.stackplot(carrier_dispatch_df.index,
    carrier_dispatch_df['onwind'],
    carrier_dispatch_df['solar'],
    carrier_dispatch_df['OCGT'],
    carrier_dispatch_df['nuclear'],
    carrier_dispatch_df['battery storage'],
    labels=techs,
    colors=carrier_colors + ['slategray'], alpha=0.85
)
ax.plot(conso_demand_ts.index, conso_demand_ts.values, color='black', lw=1.5, label='Consumption Demand')
ax.plot(total_demand_ts.index, total_demand_ts.values, color='red', lw=1.5, linestyle='--', label='Total Demand')
ax.set_title(f'Dispatch - Winter Week (January {year})')
ax.set_ylabel('Power [MW]')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_xlabel('Date')
ax.set_xlim(winter_start, winter_start + pd.Timedelta(hours=167))
fig.tight_layout()
plt.show()

# %% 
# get average lines loading
line_loading = n.lines_t.p0.abs() / n.lines.s_nom * 100
print("=== Average line loading (%) ===")
line_loading.mean()

# %%
# plot overall energy mix
# Generators
gen_cap = n.generators.p_nom_opt.copy()
bat_cap = n.storage_units.p_nom_opt

print("=== Optimal capacities (GW) ===")
print(gen_cap.div(1e3).rename(lambda x: x).to_string())
print("battery storage: {:.2f} GW  ({:.2f} GWh)".format(
    bat_cap["battery_ITA"] / 1e3,
    bat_cap["battery_ITA"] * 2 / 1e3))

print("\n=== Annual generation (TWh) ===")
gen_energy = n.generators_t.p.sum().div(1e6)
bat_discharge = n.storage_units_t.p.clip(lower=0).sum().div(1e6)
print(gen_energy.to_string())
print("battery (discharge): {:.2f} TWh".format(bat_discharge["battery_ITA"]))

total_demand_e = n.loads_t.p_set.sum().values[0] / 1e6
print("\nTotal demand: {:.2f} TWh".format(total_demand_e))
fig, ax = plt.subplots(figsize=(6, 6))
mix = gen_energy.copy()
mix = mix.sort_index()
mix = mix[mix > 0.01]
carrier_country_colors = dict(zip(
    ["OCGT_ITA", "OCGT_CHE", "OCGT_AUT", "onwind_ITA","onwind_AUT","onwind_FRA","solar_ITA","solar_CHE","nuclear_FRA","battery_ITA"],
    ["indianred","indianred","indianred","dodgerblue","dodgerblue","dodgerblue","gold",     "gold",     "purple",     "slategray"]
))
colors_pie = [carrier_country_colors.get(t, "gray") for t in mix.index]

wedges, texts, autotexts = ax.pie(
    mix, labels=mix.index, autopct="%1.1f%%",
    colors=colors_pie, startangle=90)
ax.set_title(f"Annual electricity mix — ITA-FRA-CHE-AUT {year}", fontsize=13)
plt.tight_layout()
plt.savefig("taskC_annual_mix.png", dpi=150)
plt.show()

# %%
# get power data at t=0 for question e
print("=== Node imbalance at t=0 (MW) ===")
print(n.buses_t.p.iloc[0].to_string())

print("\n=== Line flows at t=0 (MW) ===")
print(n.lines_t.p0.iloc[0].to_string())

# %%
# Economic results
total_cost = n.objective
total_demand = n.loads_t.p_set.sum().sum()
overall_lcoe = total_cost / total_demand

print("\n=== Economic Results ===")
print(f"Total system cost: {total_cost:,.2f} EUR")
print(f"Overall LCOE: {overall_lcoe:.2f} EUR/MWh")

print("\nLCOE per technology (EUR/MWh):")

# LCOE for generators
gen_lcoe = (n.generators.capital_cost * n.generators.p_nom_opt + n.generators_t.p.sum() * n.generators.marginal_cost) / n.generators_t.p.sum()
print(gen_lcoe[gen_lcoe.notna()].to_string())

# LCOE for storage units
storage_lcoe = (n.storage_units.capital_cost * n.storage_units.p_nom_opt) / n.storage_units_t.p.clip(lower=0).sum()
print(storage_lcoe[storage_lcoe.notna()].to_string())

# Bar chart of LCOE
lcoe_data = pd.concat([gen_lcoe, storage_lcoe])
lcoe_data = lcoe_data[lcoe_data.notna()]

# Since there's one LCOE per component, a bar chart is more appropriate
fig, ax = plt.subplots(figsize=(10, 6))
lcoe_data.plot(kind='bar', ax=ax)
ax.axhline(overall_lcoe, color='r', linestyle='--', label='Overall LCOE')
ax.legend()
ax.set_ylabel("LCOE [EUR/MWh]")
ax.set_title("LCOE per Technology")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %%
