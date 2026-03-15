import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypsa

year = 2020
url = f"https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_{year}.csv"
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
    "CO2 intensity": 0,
    "discount rate": 0.07,
}
costs = costs.value.unstack().fillna(defaults)

costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
costs.at["OCGT", "CO2 intensity"] = costs.at["gas", "CO2 intensity"]


def annuity(r, n):
    return r / (1.0 - 1.0 / (1.0 + r) ** n)


costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]

annuity = costs.apply(lambda x: annuity(x["discount rate"], x["lifetime"]), axis=1)
costs["capital_cost"] = (annuity + costs["FOM"] / 100) * costs["investment"]

#Capital Cost of onshore wind installation
costs.at["onwind", "capital_cost"]

#Marginal cost of onshore wind
print(costs.at["onwind", "marginal_cost"])

# print(costs.at["CCGT", "capital_cost"])
# print(costs.at["OCGT", "capital_cost"])
# print(costs.at["solar", "marginal_cost"])


# Our Country
country = "ITA"

# Wind Data
data_wind = pd.read_csv('data/onshore_wind_1979-2017.csv', sep=';', index_col=0, parse_dates=True)[country]
data_wind.index = data_wind.index.tz_localize(None)

# Solar Data
data_solar = pd.read_csv('data/pv_optimal.csv', sep=';', index_col=0, parse_dates=True)[country]
data_solar.index = data_solar.index.tz_localize(None)

# Electricity Demand
demand = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0, parse_dates=True)[country]
demand.index = demand.index.tz_localize(None)



n = pypsa.Network()

### This is for 2015 / for other years make sure to change the dates
snapshots = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00', freq='h')

n.set_snapshots(snapshots)
n.add("Bus", "ITA")
print(n.buses)

n.add('Load', "Demand", bus='ITA', p_set=demand.loc[snapshots])
#n.loads_t.p_set.plot() # Plots the demand

n.add("Generator",
      "Wind_CF",
      bus="ITA",
      p_max_pu=data_wind.loc[snapshots],
      capital_cost=costs.at["onwind", "capital_cost"],
      marginal_cost=costs.at["onwind", "marginal_cost"],
      p_nom_extendable=True)

n.add("Generator",
      "Solar_CF",
      bus="ITA",
      p_max_pu=data_solar.loc[snapshots],
      capital_cost=costs.at["solar", "capital_cost"],
      marginal_cost=costs.at["solar", "marginal_cost"],
      p_nom_extendable=True)

n.add("Generator",
      "CCGT",
      bus="ITA",
      p_max_pu=1,
      capital_cost=costs.at["CCGT", "capital_cost"],
      marginal_cost=costs.at["CCGT", "marginal_cost"],
      p_nom_extendable=True)

#n.generators_t.p_max_pu.plot()


n.optimize()

#print(f"The total cost is: {n.objective / 1e9:.2f}")

#Optimal Capacity of each generator

print(n.generators.p_nom_opt)

print("DONE!!!!")