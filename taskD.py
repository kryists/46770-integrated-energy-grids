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

costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]
annuity = costs.apply(lambda x: annuity(x["discount rate"], x["lifetime"]), axis=1)
costs["capital_cost"] = (annuity + costs["FOM"] / 100) * costs["investment"]

techs = ["onwind", "solar", "OCGT", "nuclear"]
for tech in techs:
    print(f"{tech}: Capital Cost = {costs.at[tech, 'capital_cost']:.2f} EUR/MW/a, Marginal Cost = {costs.at[tech, 'marginal_cost']:.2f} EUR/MWh")

# %% 
# Network Creation
# Nodes and loads
# ----------------------------
year = 2015
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
carriers = ["onwind", "solar", "OCGT"]
n.add(
    "Carrier",
    carriers,
    color=["dodgerblue", "gold", "indianred"]
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
    name="Nuclear_FRA",
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

# Lines
# ----------------------------
# Italy - France
n.add("Line",
    "ITA-FRA",bus0="ITA",bus1="FRA",
    s_nom=400,x=0.1,r=0.1,
)

# Italy - Switzerland
n.add("Line",
    "ITA-CHE",bus0="ITA",bus1="CHE",
    s_nom=400,x=0.1,r=0.1,
)

# Italy - Austria
n.add("Line",
    "ITA-AUT",bus0="ITA",bus1="AUT",
    s_nom=400,x=0.1,r=0.1,
)

# France - Switzerland
n.add("Line",
    "FRA-CHE",bus0="FRA",bus1="CHE",
    s_nom=400,x=0.1,r=0.1,
)

# %% 
# Solve
n.optimize(store_basis=True, keep_references=True)
# Network.optimize.fix_optimal_capacities :
# when a capacity expansion optimization was 
# already performed and a operational optimization should be done afterwards.

# %% Plot
G = n.graph()
pos = {
    'FRA': (-1, 1),
    'CHE': (0, 1),
    'AUT': (1, 1),
    'ITA': (0, 0)
}
plt.figure(figsize=(8, 6))
nx.draw(
    G, pos,
    with_labels=True, 
    node_color='lightblue', 
    node_size=2500, 
    font_weight='bold',
    edge_color='gray',
    width=2
)

plt.title("Italy and Neighbors", fontsize=14)
plt.show()

# %% Dispatch plot
# Get the dispatch time series for all generators
dispatch_df = n.generators_t.p
carriers_series = n.generators.carrier
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

total_demand = dem.sum(axis=1)

ax.stackplot(carrier_dispatch_df.index,
    carrier_dispatch_df['onwind'],
    carrier_dispatch_df['solar'],
    carrier_dispatch_df['OCGT'],
    carrier_dispatch_df['nuclear'],
    labels=techs,
    colors=['dodgerblue', 'gold', 'indianred'], alpha=0.85
)
ax.plot(total_demand.index, total_demand.values, color='black', lw=1.5, label='Total Demand')
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

total_demand = dem.sum(axis=1)

ax.stackplot(carrier_dispatch_df.index,
    carrier_dispatch_df['onwind'],
    carrier_dispatch_df['solar'],
    carrier_dispatch_df['OCGT'],
    carrier_dispatch_df['nuclear'],
    labels=techs,
    colors=['dodgerblue', 'gold', 'indianred'], alpha=0.85
)
ax.plot(total_demand.index, total_demand.values, color='black', lw=1.5, label='Total Demand')
ax.set_title(f'Dispatch - Winter Week (January {year})')
ax.set_ylabel('Power [MW]')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_xlabel('Date')
ax.set_xlim(winter_start, winter_start + pd.Timedelta(hours=167))
fig.tight_layout()
plt.show()
# %%
