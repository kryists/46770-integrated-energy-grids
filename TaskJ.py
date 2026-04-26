import pypsa
import numpy as np
import pandas as pd

n = pypsa.Network()

country = 'DNK'

def annuity(y, r):
    if r > 0:
      return r / (1.0 - 1.0 / (1.0 + r) ** y)
    else:
       return 1/y
    

year = 2010
n = pypsa.Network()
snapshots = pd.date_range(f'{year}-01-01 00:00Z', f'{year}-12-31 23:00Z', freq='h')
n.set_snapshots(snapshots.values)
n.add("Bus", "DNK")
 
# Demand Data
demand = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)
demand.index = pd.to_datetime(demand.index)

n.add('Load', "Demand", bus='DNK', p_set=demand[country].values)
 
# Carriers
n.add("Carrier", "Gas", co2_emissions=0.19)
n.add("Carrier", "Onshore Wind")
n.add("Carrier", "Offshore Wind")
n.add("Carrier", "Solar")

# Onshore Wind Data
data_on_wind = pd.read_csv('data/onshore_wind_1979-2017.csv', sep=';', index_col=0)
data_on_wind.index = pd.to_datetime(data_on_wind.index, utc=True)
cf_on_wind = data_on_wind[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ") for h in n.snapshots]]
capital_cost_wind = annuity(30, 0.07) * 910_000 * (1 + 0.033)
n.add("Generator", "Onshore Wind",
          bus="DNK", p_nom_extendable=True, carrier="Onshore Wind",
          capital_cost=capital_cost_wind, marginal_cost=0,
          p_max_pu=cf_on_wind.values)

# Offshore Wind Data
data_of_wind = pd.read_csv('data/offshore_wind_1979-2017.csv', sep=';', index_col=0)
data_of_wind.index = pd.to_datetime(data_of_wind.index, utc=True)
cf_of_wind = data_of_wind[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ") for h in n.snapshots]]
capital_cost_of_wind = annuity(25, 0.07) * 2_506_000 * (1 + 0.033)
n.add("Generator", "Offshore Wind",
          bus="DNK", p_nom_extendable=True, carrier="Offshore Wind",
          capital_cost=capital_cost_of_wind, marginal_cost=0,
          p_max_pu=cf_of_wind.values)

# Solar Data
data_solar = pd.read_csv('data/pv_optimal.csv', sep=';', index_col=0)
data_solar.index = pd.to_datetime(data_solar.index)
cf_solar = data_solar[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ") for h in n.snapshots]]
capital_cost_solar = annuity(25, 0.07) * 725_000 * (1 + 0.02)
n.add("Generator", "Solar",
          bus="DNK", p_nom_extendable=True, carrier="Solar",
          capital_cost=capital_cost_solar, marginal_cost=0,
          p_max_pu=cf_solar.values)


# Gas Data
capital_cost_OCGT = annuity(25, 0.07) * 560_000 * (1 + 0.033)
fuel_cost  = 30       # €/MWh_th
efficiency = 0.39
n.add("Generator", "OCGT",
          bus="DNK", carrier="Gas",
          capital_cost=capital_cost_OCGT,
          marginal_cost=fuel_cost / efficiency,
          p_nom_extendable=True)
    

 
n.optimize(solver_name='gurobi')


total_demand = n.loads_t.p["Demand"].sum()
caps_2010 = n.generators.p_nom_opt.copy()
co2_2010 = (n.generators_t.p["OCGT"].sum() / efficiency) * 0.19  # Total CO2 emissions from gas generation in tons
print(f"2010 CO2 emissions: {co2_2010:.2f} tCO2")


co2_cap = co2_2010 * 0.1  # 90% reduction from 2010 levels
# print(f"Marginal Cost: {n.objective / n.loads_t.p.sum().values[0]:.2f} €/MWh")
# print("Optimal capacities [MW]:")
# print(n.generators.p_nom_opt)
# print(f"Total yearly demand: {total_demand:.2f} MWh")



n_2050 = pypsa.Network()

# Snapshots (same year structure)
n_2050.set_snapshots(snapshots.values)

# Bus
n_2050.add("Bus", "DNK")


n_2050.add(
    'Load',
    "Demand",
    bus='DNK',
    p_set=1.5 * demand[country].values   
)

# ------------------ Carriers ------------------
n_2050.add("Carrier", "Gas", co2_emissions=0.19)
n_2050.add("Carrier", "Onshore Wind")
n_2050.add("Carrier", "Offshore Wind")
n_2050.add("Carrier", "Solar")

# ------------------ Onshore Wind (FIXED) ------------------
n_2050.add(
    "Generator",
    "Onshore Wind",
    bus="DNK",
    carrier="Onshore Wind",
    p_nom=caps_2010["Onshore Wind"],    
    p_nom_extendable=False,              
    capital_cost=capital_cost_wind,
    marginal_cost=0,
    p_max_pu=cf_on_wind.values
)

# ------------------ Offshore Wind ------------------
n_2050.add(
    "Generator",
    "Offshore Wind",
    bus="DNK",
    carrier="Offshore Wind",
    p_nom_extendable=True,
    capital_cost=capital_cost_of_wind,
    marginal_cost=0,
    p_max_pu=cf_of_wind.values
)

# ------------------ Solar ------------------
n_2050.add(
    "Generator",
    "Solar",
    bus="DNK",
    carrier="Solar",
    p_nom_extendable=True,
    capital_cost=capital_cost_solar,
    marginal_cost=0,
    p_max_pu=cf_solar.values
)

# ------------------ Gas ------------------
n_2050.add(
    "Generator",
    "OCGT",
    bus="DNK",
    carrier="Gas",
    p_nom_extendable=True,
    capital_cost=capital_cost_OCGT,
    marginal_cost=fuel_cost / efficiency
)

n_2050.add(
   "GlobalConstraint",
   "CO2Limit",
   carrier_attribute="co2_emissions",
   sense="<=",
   constant=co2_cap)

# ------------------ Optimize ------------------
n_2050.optimize(solver_name='gurobi')

# ------------------ Results ------------------
caps_2050 = n_2050.generators.p_nom_opt.copy()

print("\n--- Capacity Comparison (GW) ---")
comparison = pd.DataFrame({
    "2010": caps_2010 / 1000,
    "2050": caps_2050 / 1000
})
print(comparison)

print(f"\n2010 CO2 emissions: {co2_2010:.2f} tCO2")
print(f"\n2050 CO2 emissions: {co2_cap:.2f} tCO2")

print(f"\nMarginal Cost {year}: {n.objective / n.loads_t.p.sum().values[0]:.2f} €/MWh")
print(f"\nMarginal Cost 2050: {n_2050.objective / n_2050.loads_t.p.sum().values[0]:.2f} €/MWh")
