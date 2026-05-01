# %% ============================================================
# IEG Project Part 2 - Question g
# Simple linear gas-network extension of the interconnected model
#
# Main idea:
# - Electricity network: same as Part 1(d)
# - Gas network: one gas bus per country
# - Gas pipelines: PyPSA Links with fixed capacities and simple efficiencies
# - Gas-to-power: replace OCGT generators by gas turbine Links
# - Run 2 cases: CH4 and H2
#
# This is the linear approach from the lecture / exercises:
# no Weymouth, no pressure variables, no dynamic gas flow.
# ================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa

# -----------------------------
# SETTINGS
# -----------------------------
YEAR = 2010
KEEP_BATTERY = True
SOLVER_NAME = "highs"   # use "gurobi" if installed and preferred
RESULTS_DIR = "results_g"
os.makedirs(RESULTS_DIR, exist_ok=True)

COUNTRIES = ["ITA", "FRA", "CHE", "AUT"]
ELEC_LINES = {
    "ITA-FRA": ("ITA", "FRA", 3000),
    "ITA-CHE": ("ITA", "CHE", 4000),
    "ITA-AUT": ("ITA", "AUT", 1200),
    "FRA-CHE": ("FRA", "CHE", 3500),
}

# Gas pipeline lengths [km]
# (simple assumption: same topology as the electricity network)
H2_PIPELINE_CAPACITY = 3191  # MW, from Week 7 Problem 7.2

GAS_PIPELINES = {
    "ITA-FRA": ("ITA", "FRA", 500, H2_PIPELINE_CAPACITY),
    "ITA-CHE": ("ITA", "CHE", 300, H2_PIPELINE_CAPACITY),
    "ITA-AUT": ("ITA", "AUT", 450, H2_PIPELINE_CAPACITY),
    "FRA-CHE": ("FRA", "CHE", 250, H2_PIPELINE_CAPACITY),
}

# Electricity line reactance / resistance (same as your part 1d)
X_LINE = 0.1
R_LINE = 0.01

# Gas pipeline diameter assumption for all pipelines
PIPE_DIAMETER_M = 0.6   # 600 mm, like in Problem 7.1

# Gas source assumptions (very simple)
# CH4 cost uses PyPSA technology-data "gas fuel" if available
# H2 cost is an assumption here and should be discussed in the report
H2_SOURCE_MARGINAL_COST = 85.0   # EUR/MWh_th (editable)
LOCAL_BACKUP_PREMIUM = 35.0      # extra cost for local backup gas supply

# Gas turbine (generic gas-to-power)
GT_EFFICIENCY = 0.39   # simple generic electrical efficiency
# You can also set GT_EFFICIENCY = costs.at["OCGT","efficiency"] later if preferred

# Battery assumptions (same idea as your old script)
BATTERY_CAPITAL_COST = 24_678 + 2 * 12_894   # EUR/MW/a
BATTERY_MAX_HOURS = 2

# -----------------------------
# Helpers
# -----------------------------
def annuity(r, n):
    return r / (1.0 - 1.0 / (1.0 + r) ** n)

def get_hourly_values(df, column, snapshots):
    """
    Safely select one year of hourly data.
    Works whether the CSV time column is string or datetime.
    """
    df = df.copy()

    if "utc_time" in df.columns:
        df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True).dt.tz_localize(None)
        df = df.set_index("utc_time")

    values = df[column].reindex(snapshots)

    if values.isna().any():
        raise ValueError(
            f"Missing data for {column}. Check if the file contains year {snapshots[0].year}."
        )

    return values.values

def load_timeseries():
    data_solar = pd.read_csv("data/pv_optimal.csv", sep=";")
    data_wind = pd.read_csv("data/onshore_wind_1979-2017.csv", sep=";")
    data_el = pd.read_csv("data/electricity_demand.csv", sep=";")

    return data_solar, data_wind, data_el

def load_costs(year_economic_data=2020):
    url = f"https://raw.githubusercontent.com/PyPSA/technology-data/v0.11.0/outputs/costs_{year_economic_data}.csv"
    costs = pd.read_csv(url, index_col=[0, 1])

    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.unit = costs.unit.str.replace("/kW", "/MW", regex=False)

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

    # Make OCGT fuel cost use the generic gas fuel value
    costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]

    annuity_applied = costs.apply(lambda x: annuity(x["discount rate"], x["lifetime"]), axis=1)

    for tech in ["onwind", "solar", "OCGT", "nuclear"]:
        costs.at[tech, "marginal_cost"] = (
            costs.at[tech, "VOM"] + costs.at[tech, "fuel"] / costs.at[tech, "efficiency"]
        )
        costs.at[tech, "capital_cost"] = (
            annuity_applied[tech] + costs.at[tech, "FOM"] / 100
        ) * costs.at[tech, "investment"]

    return costs

def gas_capacity_mw(gas_case, D=0.6, T=298.15):
    """
    Very simple capacity estimate based on the lecture exercise idea:
    q = mdot * e ,   mdot = rho * A * u
    rho = p * M / (Z * R * T)

    Returns capacity in MW_th for one pipeline, assuming same diameter for all.
    """
    R = 8.314  # J/mol/K
    A = np.pi * (D**2) / 4

    if gas_case == "CH4":
        p = 50e5            # 50 bar in Pa
        u = 15.0            # m/s
        M = 0.016           # kg/mol
        Z = 1.31
        e_MWh_per_kg = 50 / 3.6 / 1000   # 50 GJ/t -> MWh/kg
    elif gas_case == "H2":
        p = 40e5            # 40 bar in Pa
        u = 30.0            # m/s
        M = 0.002           # kg/mol
        Z = 1.03
        e_MWh_per_kg = 120 / 3.6 / 1000  # 120 GJ/t -> MWh/kg
    else:
        raise ValueError("gas_case must be 'CH4' or 'H2'")

    rho = p * M / (Z * R * T)         # kg/m^3
    mdot = rho * A * u                # kg/s
    q_MW = mdot * e_MWh_per_kg * 3600 # MW_th
    return q_MW

def pipeline_efficiency(length_km):
    """
    Simple compressor-loss approximation from Problem 7.1(c):
    2% of energy flow per 1000 km.
    """
    loss_fraction = 0.02 * length_km / 1000
    return max(0.0, 1.0 - loss_fraction)

def get_week_slice(year, season="summer"):
    if season == "summer":
        start = pd.Timestamp(f"{year}-07-06")
    else:
        start = pd.Timestamp(f"{year}-01-06")
    end = start + pd.Timedelta(hours=167)
    return start, end

# -----------------------------
# Build model
# -----------------------------
def build_network(gas_case="H2", keep_battery=True):
    data_solar, data_wind, data_el = load_timeseries()
    costs = load_costs()

    n = pypsa.Network()
    snapshots = pd.date_range(
        f"{YEAR}-01-01 00:00",
        f"{YEAR}-12-31 23:00",
        freq="h"
    )
    n.set_snapshots(snapshots)

    # -------------------------
    # Carriers
    # -------------------------
    carriers = [
        "onwind", "solar", "nuclear",
        "battery storage",
        f"{gas_case}_source",
        f"{gas_case}_backup",
        f"{gas_case}_pipeline",
        f"{gas_case}_to_power"
    ]
    n.add("Carrier", carriers)

    # -------------------------
    # Electricity buses
    # -------------------------
    positions = {
        "ITA": (0, 0),
        "FRA": (-1, 1),
        "CHE": (0, 1),
        "AUT": (1, 1),
    }
    for c in COUNTRIES:
        n.add("Bus", c, carrier="AC", v_nom=400, x=positions[c][0], y=positions[c][1])

    # Loads
    for c in COUNTRIES:
        n.add(
            "Load",
            f"demand_{c}",
            bus=c,
            p_set=data_el[c].values[:len(snapshots)]
        )

    # -------------------------
    # Generators: RES + nuclear
    # -------------------------
    # Italy
    n.add(
        "Generator", "onwind_ITA",
        bus="ITA", carrier="onwind",
        p_max_pu=get_hourly_values(data_wind, "ITA", snapshots),
        capital_cost=costs.at["onwind", "capital_cost"],
        marginal_cost=costs.at["onwind", "marginal_cost"],
        p_nom_extendable=True,
    )
    n.add(
        "Generator", "solar_ITA",
        bus="ITA", carrier="solar",
        p_max_pu=get_hourly_values(data_solar, "ITA", snapshots),
        capital_cost=costs.at["solar", "capital_cost"],
        marginal_cost=costs.at["solar", "marginal_cost"],
        p_nom_extendable=True,
    )

    # France
    n.add(
        "Generator", "nuclear_FRA",
        bus="FRA", carrier="nuclear",
        capital_cost=costs.at["nuclear", "capital_cost"],
        marginal_cost=costs.at["nuclear", "marginal_cost"],
        p_nom_extendable=True,
    )
    n.add(
        "Generator", "onwind_FRA",
        bus="FRA", carrier="onwind",
        p_max_pu=get_hourly_values(data_wind, "FRA", snapshots),
        capital_cost=costs.at["onwind", "capital_cost"],
        marginal_cost=costs.at["onwind", "marginal_cost"],
        p_nom_extendable=True,
    )

    # Austria
    n.add(
        "Generator", "onwind_AUT",
        bus="AUT", carrier="onwind",
        p_max_pu=get_hourly_values(data_wind, "AUT", snapshots),
        capital_cost=costs.at["onwind", "capital_cost"],
        marginal_cost=costs.at["onwind", "marginal_cost"],
        p_nom_extendable=True,
    )

    # Switzerland
    n.add(
        "Generator", "solar_CHE",
        bus="CHE", carrier="solar",
        p_max_pu=get_hourly_values(data_solar, "CHE", snapshots),
        capital_cost=costs.at["solar", "capital_cost"],
        marginal_cost=costs.at["solar", "marginal_cost"],
        p_nom_extendable=True,
    )

    # -------------------------
    # Battery in Italy (optional)
    # -------------------------
    if keep_battery:
        eta_battery = 0.96
        n.add(
            "StorageUnit", "battery_ITA",
            bus="ITA", carrier="battery storage",
            max_hours=BATTERY_MAX_HOURS,
            capital_cost=BATTERY_CAPITAL_COST,
            efficiency_store=eta_battery,
            efficiency_dispatch=eta_battery,
            p_nom_extendable=True,
            cyclic_state_of_charge=True,
        )

    # -------------------------
    # Electricity lines
    # -------------------------
    for line_name, (b0, b1, s_nom) in ELEC_LINES.items():
        n.add(
            "Line", line_name,
            bus0=b0, bus1=b1,
            s_nom=s_nom,
            x=X_LINE, r=R_LINE,
        )

    # -------------------------
    # Gas buses
    # -------------------------
    for c in COUNTRIES:
        n.add("Bus", f"gas_{c}", carrier=gas_case)

    # -------------------------
    # Gas supply
    # -------------------------
    cheap_gas_cost = H2_SOURCE_MARGINAL_COST

    # One cheap import hub at FRA to force pipeline usage
    n.add(
        "Generator", f"{gas_case}_hub_FRA",
        bus="gas_FRA",
        carrier=f"{gas_case}_source",
        p_nom_extendable=True,
        capital_cost=0.0,
        marginal_cost=cheap_gas_cost,
    )

    # Expensive local backup gas in every node, to keep feasibility
    for c in COUNTRIES:
        n.add(
            "Generator", f"{gas_case}_backup_{c}",
            bus=f"gas_{c}",
            carrier=f"{gas_case}_backup",
            p_nom_extendable=True,
            capital_cost=0.0,
            marginal_cost=cheap_gas_cost + LOCAL_BACKUP_PREMIUM,
        )

    # -------------------------
    # Gas pipelines (linear approach)
    # -------------------------
    q_pipe = H2_PIPELINE_CAPACITY

    for pipe_name, (c0, c1, length_km, p_nom) in GAS_PIPELINES.items():
        eff = pipeline_efficiency(length_km)

        n.add(
            "Link", f"pipe_{pipe_name}_fw",
            bus0=f"gas_{c0}",
            bus1=f"gas_{c1}",
            carrier="H2_pipeline",
            p_nom=p_nom,
            efficiency=eff,
            marginal_cost=0.0,
        )

        n.add(
            "Link", f"pipe_{pipe_name}_bw",
            bus0=f"gas_{c1}",
            bus1=f"gas_{c0}",
            carrier="H2_pipeline",
            p_nom=p_nom,
            efficiency=eff,
            marginal_cost=0.0,
        )

    # -------------------------
    # Gas-to-power conversion
    # Replace OCGTs by links that consume gas and produce electricity
    # -------------------------
    # Important:
    # Link p_nom is on bus0 side (fuel input). Since OCGT capex is usually per MW_el,
    # convert it to a cost per MW_th input by multiplying by efficiency.
    ocgt_capital_cost_input_side = costs.at["OCGT", "capital_cost"] * GT_EFFICIENCY

    for c in COUNTRIES:
        n.add(
            "Link", f"GT_{c}",
            bus0=f"gas_{c}",
            bus1=c,
            carrier=f"{gas_case}_to_power",
            efficiency=GT_EFFICIENCY,
            p_nom_extendable=True,
            capital_cost=ocgt_capital_cost_input_side,
            marginal_cost=0.0,
        )

    return n, costs, q_pipe

# -----------------------------
# Results / plots
# -----------------------------
def make_dispatch_df(n, gas_case):
    """
    Returns electricity-side dispatch by technology.
    """
    dispatch = pd.DataFrame(index=n.snapshots)

    # Generators
    for g in n.generators.index:
        carrier = n.generators.at[g, "carrier"]
        if carrier in ["onwind", "solar", "nuclear"]:
            dispatch[g] = n.generators_t.p[g]

    # Gas turbines: electricity produced at bus1 = -p1
    gt_links = [x for x in n.links.index if x.startswith("GT_")]
    for gt in gt_links:
        dispatch[gt] = -n.links_t.p1[gt]

    # Battery discharge only
    if "battery_ITA" in n.storage_units.index:
        dispatch["battery_ITA"] = n.storage_units_t.p["battery_ITA"].clip(lower=0)

    # Grouped dispatch
    group_map = {}
    for col in dispatch.columns:
        if col.startswith("onwind"):
            group_map[col] = "onwind"
        elif col.startswith("solar"):
            group_map[col] = "solar"
        elif col.startswith("nuclear"):
            group_map[col] = "nuclear"
        elif col.startswith("GT_"):
            group_map[col] = f"{gas_case}_to_power"
        elif col == "battery_ITA":
            group_map[col] = "battery storage"
        else:
            group_map[col] = "other"

    return dispatch.T.groupby(group_map).sum().T

def plot_dispatch_week(n, gas_case, season="summer"):
    dispatch = make_dispatch_df(n, gas_case)

    start, end = get_week_slice(YEAR, season)
    sl = dispatch.loc[start:end]

    total_electric_load = pd.DataFrame({
        c: n.loads_t.p[f"demand_{c}"] for c in COUNTRIES
    }).sum(axis=1).loc[start:end]

    if "battery_ITA" in n.storage_units.index:
        charging = n.storage_units_t.p["battery_ITA"].clip(upper=0).abs().loc[start:end]
        total_demand = total_electric_load + charging
    else:
        total_demand = total_electric_load.copy()

    order = [x for x in ["onwind", "solar", "nuclear", f"{gas_case}_to_power", "battery storage"] if x in sl.columns]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(
        sl.index,
        *[sl[c].values for c in order],
        labels=order,
        alpha=0.85
    )
    ax.plot(total_electric_load.index, total_electric_load.values, color="black", lw=1.4, label="Electric demand")
    ax.plot(total_demand.index, total_demand.values, color="red", lw=1.2, ls="--", label="Demand incl. battery charging")
    ax.set_xlim(start, end)
    ax.set_ylabel("Power [MW]")
    ax.set_title(f"{gas_case} case - Electricity dispatch ({season} week)")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"dispatch_{gas_case}_{season}.png"), dpi=150)
    plt.show()

def plot_capacity_mix(n, gas_case):
    rows = []

    # Generators
    for g in n.generators.index:
        rows.append({
            "component": g,
            "carrier": n.generators.at[g, "carrier"],
            "capacity_MW": n.generators.at[g, "p_nom_opt"]
        })

    # Gas turbines: electrical output capacity = efficiency * p_nom_opt
    gt_links = [x for x in n.links.index if x.startswith("GT_")]
    for gt in gt_links:
        rows.append({
            "component": gt,
            "carrier": f"{gas_case}_to_power",
            "capacity_MW": n.links.at[gt, "p_nom_opt"] * n.links.at[gt, "efficiency"]
        })

    # Battery
    if "battery_ITA" in n.storage_units.index:
        rows.append({
            "component": "battery_ITA",
            "carrier": "battery storage",
            "capacity_MW": n.storage_units.at["battery_ITA", "p_nom_opt"]
        })

    df = pd.DataFrame(rows)
    grouped = df.groupby("carrier")["capacity_MW"].sum().sort_values(ascending=False) / 1e3

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.plot(kind="bar", ax=ax)
    ax.set_ylabel("Capacity [GW]")
    ax.set_title(f"{gas_case} case - Optimal installed capacities")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"capacities_{gas_case}.png"), dpi=150)
    plt.show()

def plot_transport_bars(n, gas_case):
    # Electricity lines
    line_transport = n.lines_t.p0.abs().sum() / 1e6  # TWh (MW * h / 1e6)

    # Gas pipelines: only pipeline links, use input-side energy
    pipe_links = [x for x in n.links.index if x.startswith("pipe_")]
    gas_transport = n.links_t.p0[pipe_links].abs().sum() / 1e6  # TWh_th

    # Aggregate reverse/forward into one physical corridor
    gas_corridor = {}
    for link_name, val in gas_transport.items():
        name = link_name.replace("pipe_", "").replace("_fw", "").replace("_bw", "")
        gas_corridor[name] = gas_corridor.get(name, 0.0) + val
    gas_corridor = pd.Series(gas_corridor).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    line_transport.sort_values(ascending=False).plot(kind="bar", ax=axes[0])
    axes[0].set_title(f"{gas_case} case - Electricity line transport")
    axes[0].set_ylabel("Annual transported energy [TWh]")

    gas_corridor.plot(kind="bar", ax=axes[1])
    axes[1].set_title(f"{gas_case} case - Gas pipeline transport")
    axes[1].set_ylabel("Annual transported energy [TWh_th]")

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"transport_bars_{gas_case}.png"), dpi=150)
    plt.show()

def summarize_results(n, gas_case, q_pipe):
    summary = {}

    summary["gas_case"] = gas_case
    summary["pipeline_capacity_MW_each"] = q_pipe
    summary["objective_EUR"] = n.objective

    # Electricity line transport
    summary["electricity_line_transport_TWh"] = float(n.lines_t.p0.abs().sum().sum() / 1e6)

    # Gas pipeline transport
    pipe_links = [x for x in n.links.index if x.startswith("pipe_")]
    summary["gas_pipeline_transport_TWh"] = float(n.links_t.p0[pipe_links].abs().sum().sum() / 1e6)

    # Total demand
    summary["electricity_demand_TWh"] = float(n.loads_t.p_set.sum().sum() / 1e6)

    # Generation by carrier
    dispatch = make_dispatch_df(n, gas_case)
    annual_dispatch = dispatch.sum() / 1e6  # TWh
    for tech, val in annual_dispatch.items():
        summary[f"gen_{tech}_TWh"] = float(val)

    # Gas source usage
    gas_source_gens = [g for g in n.generators.index if n.generators.at[g, "carrier"] in [f"{gas_case}_source", f"{gas_case}_backup"]]
    gas_supply = n.generators_t.p[gas_source_gens].sum() / 1e6
    for comp, val in gas_supply.items():
        summary[f"supply_{comp}_TWh"] = float(val)

    return summary

def save_main_tables(n, gas_case):
    # Optimal capacities
    cap_rows = []

    for g in n.generators.index:
        cap_rows.append({
            "component": g,
            "type": "Generator",
            "carrier": n.generators.at[g, "carrier"],
            "capacity_MW": n.generators.at[g, "p_nom_opt"]
        })

    for l in n.links.index:
        p_nom_opt = n.links.at[l, "p_nom_opt"] if "p_nom_opt" in n.links.columns else n.links.at[l, "p_nom"]
        cap_rows.append({
            "component": l,
            "type": "Link",
            "carrier": n.links.at[l, "carrier"],
            "capacity_MW_input_side": p_nom_opt
        })

    if len(n.storage_units):
        for s in n.storage_units.index:
            cap_rows.append({
                "component": s,
                "type": "StorageUnit",
                "carrier": n.storage_units.at[s, "carrier"],
                "capacity_MW": n.storage_units.at[s, "p_nom_opt"]
            })

    pd.DataFrame(cap_rows).to_csv(os.path.join(RESULTS_DIR, f"capacities_table_{gas_case}.csv"), index=False)

    # Annual flows
    n.lines_t.p0.to_csv(os.path.join(RESULTS_DIR, f"electricity_line_flows_{gas_case}.csv"))
    pipe_links = [x for x in n.links.index if x.startswith("pipe_")]
    n.links_t.p0[pipe_links].to_csv(os.path.join(RESULTS_DIR, f"gas_pipeline_flows_{gas_case}.csv"))


# -----------------------------
# Run H2 case only
# -----------------------------
gas_case = "H2"

print(f"\n==================== Running case: {gas_case} ====================")

n, costs, q_pipe = build_network(gas_case=gas_case, keep_battery=KEEP_BATTERY)
n.optimize(solver_name=SOLVER_NAME)

# Save tables
save_main_tables(n, gas_case)

# Plots
plot_dispatch_week(n, gas_case, season="winter")
plot_dispatch_week(n, gas_case, season="summer")
plot_capacity_mix(n, gas_case)
plot_transport_bars(n, gas_case)

# Summary
summary = summarize_results(n, gas_case, q_pipe)
summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(RESULTS_DIR, "summary_H2.csv"), index=False)

print("\n==================== H2 Summary ====================")
print(summary_df[[
    "gas_case",
    "pipeline_capacity_MW_each",
    "objective_EUR",
    "electricity_demand_TWh",
    "electricity_line_transport_TWh",
    "gas_pipeline_transport_TWh"
]].to_string(index=False))

# -----------------------------
# H2 electricity vs gas transport plot
# -----------------------------
elec_twh = summary["electricity_line_transport_TWh"]
gas_twh = summary["gas_pipeline_transport_TWh"]

transport_comparison = pd.Series({
    "Electricity lines": elec_twh,
    "H2 pipelines": gas_twh
})

fig, ax = plt.subplots(figsize=(6, 4))
transport_comparison.plot(kind="bar", ax=ax)

ax.set_title("H2 case: electricity vs hydrogen transport")
ax.set_ylabel("Annual transported energy [TWh]")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "H2_electricity_vs_gas_transport.png"), dpi=150)
plt.show()

# -----------------------------
# Simple printed conclusion
# -----------------------------
print("\n==================== Quick discussion helper ====================")

if gas_twh > elec_twh:
    dominant = "hydrogen pipeline network"
else:
    dominant = "electricity network"

print(f"H2 case:")
print(f"  - Electricity line transport: {elec_twh:.2f} TWh")
print(f"  - H2 pipeline transport:      {gas_twh:.2f} TWh")
print(f"  - Network transporting more energy: {dominant}")