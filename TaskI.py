# % ============================================================
# IEG Project Part 2 - Question i
# Sector-coupled model: Electricity + H2 Gas + Transport
#
# Extension of taskH:
# - Electricity network: 4 countries (ITA, FRA, CHE, AUT)
# - H2 gas network: linear pipelines, same topology
# - Transport sector: BEV (smart charging) + FCEV (H2 load)
# - CO2 constraint: 10% reduction from taskH non-binding baseline
# - Snapshots: 4 representative weeks (1 per season) for speed
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
YEAR         = 2015
KEEP_BATTERY = True
SOLVER_NAME  = "gurobi"

COUNTRIES = ["ITA", "FRA", "CHE", "AUT"]

ELEC_LINES = {
    "ITA-FRA": ("ITA", "FRA", 3000),
    "ITA-CHE": ("ITA", "CHE", 4000),
    "ITA-AUT": ("ITA", "AUT", 1200),
    "FRA-CHE": ("FRA", "CHE", 3500),
}
GAS_PIPELINES = {
    "ITA-FRA": ("ITA", "FRA", 500),
    "ITA-CHE": ("ITA", "CHE", 300),
    "ITA-AUT": ("ITA", "AUT", 450),
    "FRA-CHE": ("FRA", "CHE", 250),
}

X_LINE = 0.1
R_LINE = 0.01
PIPE_DIAMETER_M = 0.6

GAS_CASE             = "H2"
H2_SOURCE_MARGINAL_COST = 85.0   # EUR/MWh_th
LOCAL_BACKUP_PREMIUM    = 35.0   # EUR/MWh_th
GT_EFFICIENCY           = 0.39

BATTERY_CAPITAL_COST = 24_678 + 2 * 12_894  # EUR/MW/a
BATTERY_MAX_HOURS    = 2

reduction_target_percent = 10.0
CO2_LIMIT_TONNES = 61_778_654.335028 * (1 - reduction_target_percent / 100)

CO2_for_H2_t   = 10.0
e_MWh_per_kg_H2 = 120 / 3.6
CO2_FOR_H2_MWH = CO2_for_H2_t / e_MWh_per_kg_H2   # tCO2/MWh_th

# ------------------------------------------------------------------
# TRANSPORT PARAMETERS
# ------------------------------------------------------------------
CAR_FLEET_M = {"ITA": 37.4, "FRA": 32.5, "CHE": 4.5, "AUT": 4.9}  # millions
BEV_SHARE   = 0.30
FCEV_SHARE  = 0.10
BEV_KWH_PER_100KM  = 20.0
FCEV_KWH_PER_100KM = 33.33   # ~1 kg H2 per 100 km
KM_PER_YEAR_PER_CAR = 15_000
BEV_CHARGE_KW = 7.4   # kW per car, ~20% plugged in simultaneously
FLEX_H = 6            # hours of smart-charging flexibility

# ------------------------------------------------------------------
# REPRESENTATIVE SNAPSHOTS  (4 weeks × 168 h = 672 h)
# Weight = 8760/672 ≈ 13.04 so annual totals scale correctly
# ------------------------------------------------------------------
SEASON_STARTS = {
    "winter": pd.Timestamp(f"{YEAR}-01-06"),
    "spring": pd.Timestamp(f"{YEAR}-04-06"),
    "summer": pd.Timestamp(f"{YEAR}-07-06"),
    "autumn": pd.Timestamp(f"{YEAR}-10-06"),
}
WEEK_WEIGHT = 8760 / (4 * 168)   # ≈ 13.04  hours weight per snapshot

def make_snapshots():
    """4 representative weeks, one per season."""
    periods = []
    for start in SEASON_STARTS.values():
        periods.append(pd.date_range(start, periods=168, freq="h"))
    return periods[0].append(periods[1:])

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def annuity(r, n):
    return r / (1.0 - 1.0 / (1.0 + r) ** n)

def get_hourly_values(df, column, snapshots):
    df = df.copy()
    if "utc_time" in df.columns:
        df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True).dt.tz_localize(None)
        df = df.set_index("utc_time")
    values = df[column].reindex(snapshots)
    if values.isna().any():
        raise ValueError(f"Missing data for '{column}'.")
    return values.values

def load_timeseries():
    solar = pd.read_csv("data/pv_optimal.csv", sep=";")
    wind  = pd.read_csv("data/onshore_wind_1979-2017.csv", sep=";")
    elec  = pd.read_csv("data/electricity_demand.csv", sep=";")
    return solar, wind, elec

def load_costs(year=2020):
    url = f"https://raw.githubusercontent.com/PyPSA/technology-data/v0.11.0/outputs/costs_{year}.csv"
    costs = pd.read_csv(url, index_col=[0, 1])
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.unit = costs.unit.str.replace("/kW", "/MW", regex=False)
    defaults = {"FOM": 0, "VOM": 0, "efficiency": 1, "fuel": 0,
                "investment": 0, "lifetime": 25, "discount rate": 0.07}
    costs = costs.value.unstack().fillna(defaults)
    costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
    costs["co2_emissions"] = 0.0
    ann = costs.apply(lambda x: annuity(x["discount rate"], x["lifetime"]), axis=1)
    for tech in ["onwind", "solar", "OCGT", "nuclear"]:
        costs.at[tech, "marginal_cost"] = (
            costs.at[tech, "VOM"] + costs.at[tech, "fuel"] / costs.at[tech, "efficiency"])
        costs.at[tech, "capital_cost"] = (
            ann[tech] + costs.at[tech, "FOM"] / 100) * costs.at[tech, "investment"]
    return costs

def gas_capacity_mw(D=0.6, T=298.15):
    R = 8.314; A = np.pi * D**2 / 4
    p = 40e5; u = 30.0; M = 0.002; Z = 1.03
    e_MWh_per_kg = 120 / 3.6 / 1000
    rho = p * M / (Z * R * T)
    return rho * A * u * e_MWh_per_kg * 3600

def pipeline_efficiency(length_km):
    return max(0.0, 1.0 - 0.02 * length_km / 1000)

def compute_transport_demands():
    bev_mw = {}; fcev_mw = {}
    for c, fleet_m in CAR_FLEET_M.items():
        fleet = fleet_m * 1e6
        bev_mw[c]  = fleet * BEV_SHARE  * KM_PER_YEAR_PER_CAR / 100 * BEV_KWH_PER_100KM  / 1e3 / 8760
        fcev_mw[c] = fleet * FCEV_SHARE * KM_PER_YEAR_PER_CAR / 100 * FCEV_KWH_PER_100KM / 1e3 / 8760
    return bev_mw, fcev_mw

# ------------------------------------------------------------------
# BUILD NETWORK
# ------------------------------------------------------------------
def build_network():
    solar, wind, elec = load_timeseries()
    costs = load_costs()
    snapshots = make_snapshots()

    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.snapshot_weightings[:] = WEEK_WEIGHT   # scale to full year

    # --- Carriers ---
    for c in ["onwind", "solar", "nuclear", "battery storage", "BEV_charging"]:
        n.add("Carrier", c)
    n.add("Carrier", f"{GAS_CASE}_source",   co2_emissions=CO2_FOR_H2_MWH)
    n.add("Carrier", f"{GAS_CASE}_backup",   co2_emissions=CO2_FOR_H2_MWH)
    n.add("Carrier", f"{GAS_CASE}_pipeline")
    n.add("Carrier", f"{GAS_CASE}_to_power", co2_emissions=0.0)

    # --- Electricity buses ---
    positions = {"ITA": (0,0), "FRA": (-1,1), "CHE": (0,1), "AUT": (1,1)}
    for c in COUNTRIES:
        n.add("Bus", c, carrier="AC", v_nom=400,
              x=positions[c][0], y=positions[c][1])

    # --- Electricity loads ---
    for c in COUNTRIES:
        n.add("Load", f"demand_{c}", bus=c,
              p_set=get_hourly_values(elec, c, snapshots))

    # --- Generators ---
    n.add("Generator", "onwind_ITA", bus="ITA", carrier="onwind",
          p_max_pu=get_hourly_values(wind, "ITA", snapshots),
          capital_cost=costs.at["onwind","capital_cost"],
          marginal_cost=costs.at["onwind","marginal_cost"], p_nom_extendable=True)
    n.add("Generator", "solar_ITA", bus="ITA", carrier="solar",
          p_max_pu=get_hourly_values(solar, "ITA", snapshots),
          capital_cost=costs.at["solar","capital_cost"],
          marginal_cost=costs.at["solar","marginal_cost"], p_nom_extendable=True)
    n.add("Generator", "nuclear_FRA", bus="FRA", carrier="nuclear",
          capital_cost=costs.at["nuclear","capital_cost"],
          marginal_cost=costs.at["nuclear","marginal_cost"], p_nom_extendable=True)
    n.add("Generator", "onwind_FRA", bus="FRA", carrier="onwind",
          p_max_pu=get_hourly_values(wind, "FRA", snapshots),
          capital_cost=costs.at["onwind","capital_cost"],
          marginal_cost=costs.at["onwind","marginal_cost"], p_nom_extendable=True)
    n.add("Generator", "onwind_AUT", bus="AUT", carrier="onwind",
          p_max_pu=get_hourly_values(wind, "AUT", snapshots),
          capital_cost=costs.at["onwind","capital_cost"],
          marginal_cost=costs.at["onwind","marginal_cost"], p_nom_extendable=True)
    n.add("Generator", "solar_CHE", bus="CHE", carrier="solar",
          p_max_pu=get_hourly_values(solar, "CHE", snapshots),
          capital_cost=costs.at["solar","capital_cost"],
          marginal_cost=costs.at["solar","marginal_cost"], p_nom_extendable=True)

    # --- Battery (Italy) ---
    if KEEP_BATTERY:
        n.add("StorageUnit", "battery_ITA", bus="ITA", carrier="battery storage",
              max_hours=BATTERY_MAX_HOURS, capital_cost=BATTERY_CAPITAL_COST,
              efficiency_store=0.96, efficiency_dispatch=0.96,
              p_nom_extendable=True, cyclic_state_of_charge=True)

    # --- Electricity lines ---
    for name, (b0, b1, s_nom) in ELEC_LINES.items():
        n.add("Line", name, bus0=b0, bus1=b1, s_nom=s_nom, x=X_LINE, r=R_LINE)

    # --- H2 buses ---
    for c in COUNTRIES:
        n.add("Bus", f"gas_{c}", carrier=GAS_CASE)

    # --- H2 supply ---
    n.add("Generator", f"{GAS_CASE}_hub_FRA", bus="gas_FRA",
          carrier=f"{GAS_CASE}_source", p_nom_extendable=True,
          capital_cost=0.0, marginal_cost=H2_SOURCE_MARGINAL_COST)
    for c in COUNTRIES:
        n.add("Generator", f"{GAS_CASE}_backup_{c}", bus=f"gas_{c}",
              carrier=f"{GAS_CASE}_backup", p_nom_extendable=True,
              capital_cost=0.0, marginal_cost=H2_SOURCE_MARGINAL_COST + LOCAL_BACKUP_PREMIUM)

    # --- H2 pipelines ---
    q_pipe = gas_capacity_mw(D=PIPE_DIAMETER_M)
    for name, (c0, c1, length_km) in GAS_PIPELINES.items():
        eff = pipeline_efficiency(length_km)
        n.add("Link", f"pipe_{name}_fw", bus0=f"gas_{c0}", bus1=f"gas_{c1}",
              carrier=f"{GAS_CASE}_pipeline", p_nom=q_pipe, efficiency=eff, marginal_cost=0.0)
        n.add("Link", f"pipe_{name}_bw", bus0=f"gas_{c1}", bus1=f"gas_{c0}",
              carrier=f"{GAS_CASE}_pipeline", p_nom=q_pipe, efficiency=eff, marginal_cost=0.0)

    # --- H2 turbines (gas-to-power) ---
    ocgt_capex_input = costs.at["OCGT","capital_cost"] * GT_EFFICIENCY
    for c in COUNTRIES:
        n.add("Link", f"GT_{c}", bus0=f"gas_{c}", bus1=c,
              carrier=f"{GAS_CASE}_to_power", efficiency=GT_EFFICIENCY,
              p_nom_extendable=True, capital_cost=ocgt_capex_input, marginal_cost=0.0)

    # ================================================================
    # TRANSPORT SECTOR
    # ================================================================
    bev_mw, fcev_mw = compute_transport_demands()

    for c in COUNTRIES:
        # BEV: fixed baseline load + small flexibility buffer (StorageUnit)
        n.add("Load", f"BEV_load_{c}", bus=c,
              p_set=bev_mw[c] * np.ones(len(snapshots)))

        fleet = CAR_FLEET_M[c] * 1e6
        bev_max_charge_mw = fleet * BEV_SHARE * BEV_CHARGE_KW / 1e6 * 0.20
        n.add("StorageUnit", f"BEV_flex_{c}", bus=c,
              carrier="BEV_charging",
              max_hours=FLEX_H,
              p_nom=bev_max_charge_mw,
              p_nom_extendable=False,
              efficiency_store=0.93, efficiency_dispatch=0.93,
              cyclic_state_of_charge=True,
              standing_loss=0.001,
              capital_cost=0.0, marginal_cost=0.0)

        # FCEV: flat H2 load on gas bus (tank-to-wheel eff ~60%)
        fcev_h2_mw = fcev_mw[c] / 0.60
        n.add("Load", f"FCEV_H2_{c}", bus=f"gas_{c}",
              p_set=fcev_h2_mw * np.ones(len(snapshots)))

    return n, costs, q_pipe, bev_mw, fcev_mw

# ------------------------------------------------------------------
# DISPATCH FRAME
# ------------------------------------------------------------------
def make_dispatch_df(n):
    dispatch = pd.DataFrame(index=n.snapshots)
    for g in n.generators.index:
        if n.generators.at[g,"carrier"] in ["onwind","solar","nuclear"]:
            dispatch[g] = n.generators_t.p[g]
    for gt in [x for x in n.links.index if x.startswith("GT_")]:
        dispatch[gt] = -n.links_t.p1[gt]
    if "battery_ITA" in n.storage_units.index:
        dispatch["battery_ITA"] = n.storage_units_t.p["battery_ITA"].clip(lower=0)
    group_map = {}
    for col in dispatch.columns:
        if   col.startswith("onwind"):  group_map[col] = "onwind"
        elif col.startswith("solar"):   group_map[col] = "solar"
        elif col.startswith("nuclear"): group_map[col] = "nuclear"
        elif col.startswith("GT_"):     group_map[col] = "H2_to_power"
        elif col == "battery_ITA":      group_map[col] = "battery storage"
        else:                           group_map[col] = "other"
    return dispatch.T.groupby(group_map).sum().T

# ------------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------------
COLORS = {"onwind":"#3A85C8","solar":"#F5A623","nuclear":"#7B68EE",
          "H2_to_power":"#E74C3C","battery storage":"#2ECC71"}

def plot_dispatch_week(n, season="summer"):
    dispatch = make_dispatch_df(n)
    start = SEASON_STARTS[season]
    end   = start + pd.Timedelta(hours=167)
    sl    = dispatch.loc[start:end]

    elec_load = sum(n.loads_t.p[f"demand_{c}"] for c in COUNTRIES).loc[start:end]
    bev_load  = sum(n.loads_t.p[f"BEV_load_{c}"] for c in COUNTRIES).loc[start:end]
    total     = elec_load + bev_load

    order  = [k for k in ["onwind","solar","nuclear","H2_to_power","battery storage"] if k in sl.columns]
    colors = [COLORS.get(k,"grey") for k in order]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(sl.index, *[sl[c].values for c in order],
                 labels=order, colors=colors, alpha=0.85)
    ax.plot(total.index, total.values, "k-", lw=1.4, label="Total demand (incl. BEV)")
    ax.set_xlim(start, end)
    ax.set_ylabel("Power [MW]")
    ax.set_title(f"Electricity dispatch – {season} week  |  Electricity + H2 + Transport")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    plt.savefig(f"pics/Task_I_dispatch_{season}.png", dpi=150)
    plt.show()

def plot_capacity_mix(n):
    rows = []
    for g in n.generators.index:
        rows.append({"carrier": n.generators.at[g,"carrier"],
                     "cap_MW":  n.generators.at[g,"p_nom_opt"]})
    for gt in [x for x in n.links.index if x.startswith("GT_")]:
        rows.append({"carrier":"H2_to_power",
                     "cap_MW": n.links.at[gt,"p_nom_opt"] * n.links.at[gt,"efficiency"]})
    if "battery_ITA" in n.storage_units.index:
        rows.append({"carrier":"battery storage",
                     "cap_MW": n.storage_units.at["battery_ITA","p_nom_opt"]})

    df = pd.DataFrame(rows)
    grouped = df.groupby("carrier")["cap_MW"].sum().sort_values(ascending=False) / 1e3
    # drop near-zero sources/backups from the capacity chart
    grouped = grouped[~grouped.index.str.contains("source|backup|pipeline|FCEV|BEV")]

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.plot(kind="bar", ax=ax,
                 color=[COLORS.get(c,"#888") for c in grouped.index])
    ax.set_ylabel("Capacity [GW]")
    ax.set_title("Optimal installed capacities  |  Electricity + H2 + Transport")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.grid(axis="y", ls="--", alpha=0.6)
    fig.tight_layout()
    plt.savefig("pics/Task_I_capacities.png", dpi=150)
    plt.show()

def plot_transport_bars(n):
    line_transport = n.lines_t.p0.abs().sum() * WEEK_WEIGHT / 1e6

    pipe_links = [x for x in n.links.index if x.startswith("pipe_")]
    gas_corridor = {}
    for lk, val in (n.links_t.p0[pipe_links].abs().sum() * WEEK_WEIGHT / 1e6).items():
        name = lk.replace("pipe_","").replace("_fw","").replace("_bw","")
        gas_corridor[name] = gas_corridor.get(name, 0.0) + val
    gas_corridor = pd.Series(gas_corridor).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    line_transport.sort_values(ascending=False).plot(kind="bar", ax=axes[0], color="#3A85C8")
    axes[0].set_title("Electricity line transport [TWh/yr]")
    axes[0].set_ylabel("TWh")

    gas_corridor.plot(kind="bar", ax=axes[1], color="#E74C3C")
    axes[1].set_title("H2 pipeline transport [TWh_th/yr]")
    axes[1].set_ylabel("TWh_th")

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
        ax.grid(axis="y", ls="--", alpha=0.6)
    fig.tight_layout()
    plt.savefig("pics/Task_I_transport_bars.png", dpi=150)
    plt.show()

def plot_sector_energy(bev_mw, fcev_mw, n):
    sectors = {}
    for c in COUNTRIES:
        elec_twh = n.loads_t.p[f"demand_{c}"].sum() * WEEK_WEIGHT / 1e6
        sectors[c] = {
            "Electricity (grid)": elec_twh,
            "BEV (electricity)":  bev_mw[c]  * 8760 / 1e6,
            "FCEV (H2 final)":    fcev_mw[c] * 8760 / 1e6,
        }
    df = pd.DataFrame(sectors).T
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind="bar", ax=ax, color=["#3A85C8","#1ABC9C","#E74C3C"])
    ax.set_ylabel("Annual energy [TWh]")
    ax.set_title("Annual demand by sector and country")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls="--", alpha=0.6)
    fig.tight_layout()
    plt.savefig("pics/Task_I_sector_energy.png", dpi=150)
    plt.show()

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
print("Building sector-coupled network ...")
n, costs, q_pipe, bev_mw, fcev_mw = build_network()

n.add("GlobalConstraint", "CO2Limit",
      carrier_attribute="co2_emissions",
      sense="<=", constant=CO2_LIMIT_TONNES)

print(f"  Snapshots : {len(n.snapshots)} ({4} representative weeks × 168 h, weight={WEEK_WEIGHT:.2f})")
print(f"  Buses     : {len(n.buses)}")
print(f"  Links     : {len(n.links)}")
print(f"  Generators: {len(n.generators)}")

bev_tot  = sum(bev_mw.values())  * 8760 / 1e6
fcev_tot = sum(fcev_mw.values()) * 8760 / 1e6
print(f"\nTransport demands  (annual equivalents):")
print(f"  BEV  total : {bev_tot:.2f} TWh_el / yr")
print(f"  FCEV total : {fcev_tot:.2f} TWh_H2(final) / yr")
for c in COUNTRIES:
    print(f"  {c}: BEV={bev_mw[c]*1e3:.1f} MW   FCEV_H2={fcev_mw[c]*1e3:.1f} MW")

print(f"\nSolving ...")
n.optimize(solver_name=SOLVER_NAME)

# --- Plots ---
plot_dispatch_week(n, season="winter")
plot_dispatch_week(n, season="summer")
plot_capacity_mix(n)
plot_transport_bars(n)
plot_sector_energy(bev_mw, fcev_mw, n)

# --- Summary printout ---
dispatch    = make_dispatch_df(n)
annual_gen  = (dispatch.sum() * WEEK_WEIGHT) / 1e6
co2_price   = -n.global_constraints.at["CO2Limit","mu"]
total_co2   = sum(
    n.generators_t.p[g].sum() * WEEK_WEIGHT
    * n.carriers.at[n.generators.at[g,"carrier"],"co2_emissions"]
    for g in n.generators.index
    if n.carriers.at[n.generators.at[g,"carrier"],"co2_emissions"] > 0
)
elec_demand_twh = sum(
    n.loads_t.p[f"demand_{c}"].sum() * WEEK_WEIGHT / 1e6 for c in COUNTRIES
)

print("\n" + "="*55)
print("RESULTS SUMMARY")
print("="*55)
print(f"  Total system cost   : {n.objective/1e9:.3f} Bn EUR/yr")
print(f"  Avg electricity cost: {n.objective/(elec_demand_twh*1e6):.1f} EUR/MWh")
print(f"  CO2 shadow price    : {co2_price:.2f} EUR/tCO2")
print(f"  Total CO2 emissions : {total_co2/1e6:.3f} Mt CO2")
print(f"  Electricity demand  : {elec_demand_twh:.1f} TWh/yr")
print(f"  BEV demand          : {bev_tot:.2f} TWh_el/yr")
print(f"  FCEV H2 demand      : {fcev_tot:.2f} TWh_H2(final)/yr")
print(f"\n  Annual generation [TWh]:")
for tech, val in annual_gen.sort_values(ascending=False).items():
    if val > 0.01:
        print(f"    {tech:20s}: {val:.2f}")
