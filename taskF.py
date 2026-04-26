"""
Task F — CO2 constraint sensitivity analysis
Italy single-country model — built on Task C (taskA.ipynb)

Investigates how the optimal capacity mix changes as the global CO2
budget is tightened from the unconstrained level down to near-zero.

Historical reference (electricity + heat, Italy):
  1990: 143 Mt CO2    Source: Our World in Data
  2023:  86.4 Mt CO2  https://ourworldindata.org/grapher/co-emissions-by-sector?country=~ITA

Note: these reference values include heat generation. The model covers
electricity only, so unconstrained model emissions will be lower than
the reference — this is expected and discussed in the report.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ── Utility (identical to Task C) ─────────────────────────────────────────────
def annuity(y, r):
    if r > 0:
        return r / (1.0 - 1.0 / (1.0 + r) ** y)
    else:
        return 1 / y

def strip_tz(s):
    """Return a timezone-naive copy of a Series/DataFrame."""
    if hasattr(s.index, 'tz') and s.index.tz is not None:
        s = s.copy()
        s.index = s.index.tz_convert(None)
    return s

# ── Parameters — exactly as in Task C (taskA.ipynb, Cell 101) ─────────────────
year    = 2010
country = "ITA"

capital_cost_wind  = annuity(30, 0.07) * 910_000  * (1 + 0.033)   # EUR/MW/a
capital_cost_solar = annuity(25, 0.07) * 425_000  * (1 + 0.030)   # EUR/MW/a
capital_cost_OCGT  = annuity(25, 0.07) * 560_000  * (1 + 0.033)   # EUR/MW/a
fuel_cost          = 21.6                                           # EUR/MWh_th
efficiency_OCGT    = 0.39
marginal_cost_OCGT = fuel_cost / efficiency_OCGT                   # EUR/MWh_el
capital_cost_battery = 24_678 + 2 * 12_894                         # EUR/MW/a
eta_battery          = 0.96

CO2_INTENSITY = 0.19   # tCO2/MWh_thermal (on Gas carrier, same as Task C)

# ── Historical references (Our World in Data, electricity + heat) ─────────────
# Source: https://ourworldindata.org/grapher/co-emissions-by-sector?country=~ITA
CO2_1990_Mt = 143.0    # MtCO2/year
CO2_2023_Mt =  86.4    # MtCO2/year
CO2_1990    = CO2_1990_Mt * 1e6   # tCO2/year
CO2_2023    = CO2_2023_Mt * 1e6   # tCO2/year

# ── Load data (identical to Task C) ───────────────────────────────────────────
demand     = pd.read_csv('data/electricity_demand.csv',     sep=';', index_col=0)
data_wind  = pd.read_csv('data/onshore_wind_1979-2017.csv', sep=';', index_col=0)
data_solar = pd.read_csv('data/pv_optimal.csv',             sep=';', index_col=0)

demand.index     = pd.to_datetime(demand.index)
data_wind.index  = pd.to_datetime(data_wind.index, utc=True)
data_solar.index = pd.to_datetime(data_solar.index)

# ── Network builder — mirrors Task C exactly ──────────────────────────────────
def build_network(co2_limit=None):
    """
    Build and solve the Italy single-country model from Task C.
    Optionally add a GlobalConstraint capping CO2 at co2_limit (tCO2/year).
    Returns the solved network.
    """
    n = pypsa.Network()
    snapshots = pd.date_range(f'{year}-01-01 00:00Z',
                              f'{year}-12-31 23:00Z', freq='h')
    n.set_snapshots(snapshots.values)
    n.add("Bus", "ITA")

    # Carriers — Gas carries CO2 intensity (same as Task C)
    n.add("Carrier", "Gas",     co2_emissions=CO2_INTENSITY)
    n.add("Carrier", "Wind")
    n.add("Carrier", "Solar")
    n.add("Carrier", "Battery")

    # Demand
    n.add('Load', "Demand", bus='ITA', p_set=demand[country].values)

    # Wind
    cf_wind = data_wind[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ")
                                  for h in n.snapshots]]
    n.add("Generator", "Wind",
          bus="ITA", carrier="Wind", p_nom_extendable=True,
          capital_cost=capital_cost_wind, marginal_cost=0,
          p_max_pu=cf_wind.values)

    # Solar
    cf_solar = data_solar[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ")
                                    for h in n.snapshots]]
    n.add("Generator", "Solar",
          bus="ITA", carrier="Solar", p_nom_extendable=True,
          capital_cost=capital_cost_solar, marginal_cost=0,
          p_max_pu=cf_solar.values)

    # OCGT — Gas carrier with CO2 intensity
    n.add("Generator", "OCGT",
          bus="ITA", carrier="Gas", p_nom_extendable=True,
          capital_cost=capital_cost_OCGT,
          marginal_cost=marginal_cost_OCGT)

    # Battery storage
    n.add("StorageUnit", "Battery",
          bus="ITA", carrier="Battery",
          max_hours=2,
          capital_cost=capital_cost_battery,
          efficiency_store=eta_battery,
          efficiency_dispatch=eta_battery,
          p_nom_extendable=True,
          cyclic_state_of_charge=True)

    # Global CO2 constraint (only when a limit is specified)
    if co2_limit is not None:
        n.add("GlobalConstraint", "co2_limit",
              sense="<=",
              carrier_attribute="co2_emissions",
              constant=co2_limit)

    n.optimize(solver_name='highs')
    return n


# =============================================================================
# STEP 1 — Run unconstrained model to find natural CO2 ceiling
# =============================================================================
print("=" * 60)
print("STEP 1: Unconstrained model (Task C baseline)")
print("=" * 60)

nc = build_network(co2_limit=None)

# Total CO2: OCGT generation (MWh_el) / efficiency x CO2 intensity (tCO2/MWh_th)
ocgt_gen_mwh = nc.generators_t.p["OCGT"].sum()                 # MWh_el
co2_free     = ocgt_gen_mwh / efficiency_OCGT * CO2_INTENSITY  # tCO2

print(f"  OCGT generation       : {ocgt_gen_mwh/1e6:.3f} TWh_el")
print(f"  Unconstrained CO2     : {co2_free/1e6:.3f} MtCO2/year  (electricity only)")
print(f"  Italy 1990 reference  : {CO2_1990_Mt:.1f} MtCO2/year  (electricity + heat)")
print(f"  Italy 2023 reference  : {CO2_2023_Mt:.1f} MtCO2/year  (electricity + heat)")
print(f"  Model emits {co2_free/CO2_1990*100:.1f}% of the 1990 reference")
print(f"  (gap expected: model covers electricity only)")
print()


# =============================================================================
# STEP 2 — Define sweep range
# =============================================================================
# Upper bound  = unconstrained model emissions
# Lower bound  = 1% of unconstrained (near-zero, fully decarbonised)
# Points denser at tight end where technology transitions happen fastest

co2_limits = np.unique(np.concatenate([
    np.linspace(0.01 * co2_free, 0.25 * co2_free, 12),  # 1-25%  tight region
    np.linspace(0.25 * co2_free, 0.70 * co2_free,  8),  # 25-70% transition
    np.linspace(0.70 * co2_free, 1.00 * co2_free,  5),  # 70-100% near-free
]))

print(f"STEP 2: Sweep — {len(co2_limits)} points from "
      f"{co2_limits[0]/1e6:.3f} to {co2_limits[-1]/1e6:.3f} MtCO2/year")
print()


# =============================================================================
# STEP 3 — Parametric sweep
# =============================================================================
print("=" * 60)
print("STEP 3: CO2 sweep")
print("=" * 60)

records = []

for i, limit in enumerate(co2_limits):
    pct = limit / co2_free * 100
    print(f"  [{i+1:02d}/{len(co2_limits)}]  "
          f"CO2 = {limit/1e6:.3f} MtCO2  ({pct:.0f}% of unconstrained)")

    try:
        n = build_network(co2_limit=limit)
        co2_price = n.global_constraints.at["co2_limit", "mu"]  # EUR/tCO2

        records.append({
            "co2_limit_Mt"  : limit / 1e6,
            "pct_free"      : pct,
            "Wind_GW"       : n.generators.at["Wind",  "p_nom_opt"] / 1e3,
            "Solar_GW"      : n.generators.at["Solar", "p_nom_opt"] / 1e3,
            "OCGT_GW"       : n.generators.at["OCGT",  "p_nom_opt"] / 1e3,
            "Battery_GW"    : n.storage_units.at["Battery", "p_nom_opt"] / 1e3,
            "Wind_TWh"      : n.generators_t.p["Wind"].sum()  / 1e6,
            "Solar_TWh"     : n.generators_t.p["Solar"].sum() / 1e6,
            "OCGT_TWh"      : n.generators_t.p["OCGT"].sum()  / 1e6,
            "Bat_TWh"       : n.storage_units_t.p["Battery"].clip(lower=0).sum() / 1e6,
            "cost_MEur"     : n.objective / 1e6,
            "co2_price_EUR" : co2_price,
        })

    except Exception as e:
        print(f"    !! Solver failed: {e}")

# Append unconstrained result (shadow price = 0 by definition)
records.append({
    "co2_limit_Mt"  : co2_free / 1e6,
    "pct_free"      : 100.0,
    "Wind_GW"       : nc.generators.at["Wind",  "p_nom_opt"] / 1e3,
    "Solar_GW"      : nc.generators.at["Solar", "p_nom_opt"] / 1e3,
    "OCGT_GW"       : nc.generators.at["OCGT",  "p_nom_opt"] / 1e3,
    "Battery_GW"    : nc.storage_units.at["Battery", "p_nom_opt"] / 1e3,
    "Wind_TWh"      : nc.generators_t.p["Wind"].sum()  / 1e6,
    "Solar_TWh"     : nc.generators_t.p["Solar"].sum() / 1e6,
    "OCGT_TWh"      : nc.generators_t.p["OCGT"].sum()  / 1e6,
    "Bat_TWh"       : nc.storage_units_t.p["Battery"].clip(lower=0).sum() / 1e6,
    "cost_MEur"     : nc.objective / 1e6,
    "co2_price_EUR" : 0.0,
})

df = (pd.DataFrame(records)
        .sort_values("co2_limit_Mt")
        .reset_index(drop=True))

print("\nSweep complete.")


# =============================================================================
# STEP 4 — Plots
# =============================================================================
COLORS = {
    "Wind"   : "steelblue",
    "Solar"  : "gold",
    "OCGT"   : "indianred",
    "Battery": "slategray",
}

x = df["co2_limit_Mt"].values

# Vertical reference lines — only drawn if they fall within x-axis range
ref_lines = {
    f"1990 ref\n({CO2_1990_Mt:.0f} Mt)": CO2_1990_Mt,
    f"2023 ref\n({CO2_2023_Mt:.0f} Mt)": CO2_2023_Mt,
    f"Unconstrained\n({co2_free/1e6:.1f} Mt)": co2_free / 1e6,
}

def add_ref_lines(ax):
    ylim = ax.get_ylim()
    for label, xval in ref_lines.items():
        if xval <= x.max() * 1.05:
            ax.axvline(xval, color="black", lw=1.1, ls="--", alpha=0.5)
            ax.text(xval, ylim[0] + 0.93 * (ylim[1] - ylim[0]),
                    label, ha="center", va="top", fontsize=7,
                    bbox=dict(fc="white", ec="none", alpha=0.7))


# ── Figure 1: Three-panel summary ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=True)
fig.suptitle(
    f"Task F — CO$_2$ constraint sensitivity\n"
    f"Italy single-country model ({year})",
    fontsize=13, y=0.99)

# Panel a: Capacity mix
ax = axes[0]
ax.stackplot(x,
             df["Wind_GW"], df["Solar_GW"],
             df["OCGT_GW"], df["Battery_GW"],
             labels=["Wind", "Solar", "OCGT", "Battery"],
             colors=[COLORS[k] for k in ["Wind","Solar","OCGT","Battery"]],
             alpha=0.85)
ax.set_ylabel("Optimal capacity (GW)")
ax.set_title("a) Optimal capacity mix vs. CO$_2$ budget")
ax.legend(loc="upper left", fontsize=9)
ax.set_xlim(left=0)
ax.grid(axis="y", alpha=0.3)
add_ref_lines(ax)

# Panel b: System cost
ax = axes[1]
ax.plot(x, df["cost_MEur"], color="darkred", lw=2, marker="o", ms=4)
ax.set_ylabel("Total system cost (M€/year)")
ax.set_title("b) Total system cost vs. CO$_2$ budget")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.set_xlim(left=0)
ax.grid(axis="y", alpha=0.3)
add_ref_lines(ax)

# Panel c: CO2 shadow price
ax = axes[2]
ax.plot(x, df["co2_price_EUR"], color="darkorange", lw=2, marker="o", ms=4)
ax.set_ylabel("CO$_2$ shadow price (€/tCO$_2$)")
ax.set_xlabel("CO$_2$ budget (MtCO$_2$/year)")
ax.set_title("c) Implied CO$_2$ price vs. CO$_2$ budget")
ax.axhspan(60, 100, alpha=0.12, color="green",
           label="EU ETS range (approx. 60-100 EUR/t, 2021-2024)")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(left=0)
ax.grid(axis="y", alpha=0.3)
add_ref_lines(ax)

fig.tight_layout()
plt.savefig("pics/taskF_co2_sensitivity.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: pics/taskF_co2_sensitivity.png")


# ── Figure 2: Annual generation mix ───────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.stackplot(x,
              df["Wind_TWh"], df["Solar_TWh"],
              df["OCGT_TWh"], df["Bat_TWh"],
              labels=["Wind", "Solar", "OCGT", "Battery discharge"],
              colors=[COLORS[k] for k in ["Wind","Solar","OCGT","Battery"]],
              alpha=0.85)
ax2.set_xlabel("CO$_2$ budget (MtCO$_2$/year)")
ax2.set_ylabel("Annual generation (TWh)")
ax2.set_title(f"Task F — Annual generation mix vs. CO$_2$ budget  (Italy {year})")
ax2.legend(loc="upper left", fontsize=9)
ax2.set_xlim(left=0)
ax2.grid(axis="y", alpha=0.3)
add_ref_lines(ax2)
fig2.tight_layout()
plt.savefig("pics/taskF_generation_mix.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: pics/taskF_generation_mix.png")


# ── Figure 3: Battery vs OCGT trade-off ───────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3_r = ax3.twinx()
ax3.plot(x, df["Battery_GW"], color=COLORS["Battery"],
         lw=2, marker="s", ms=5, label="Battery (GW)")
ax3_r.plot(x, df["OCGT_GW"], color=COLORS["OCGT"],
           lw=2, marker="^", ms=5, ls="--", label="OCGT (GW)")
ax3.set_xlabel("CO$_2$ budget (MtCO$_2$/year)")
ax3.set_ylabel("Battery capacity (GW)", color=COLORS["Battery"])
ax3_r.set_ylabel("OCGT capacity (GW)",  color=COLORS["OCGT"])
ax3.tick_params(axis='y', labelcolor=COLORS["Battery"])
ax3_r.tick_params(axis='y', labelcolor=COLORS["OCGT"])
ax3.set_title("Task F — Battery vs. OCGT trade-off as CO$_2$ is tightened")
l1, lab1 = ax3.get_legend_handles_labels()
l2, lab2 = ax3_r.get_legend_handles_labels()
ax3.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=9)
ax3.set_xlim(left=0)
ax3.grid(axis="y", alpha=0.3)
fig3.tight_layout()
plt.savefig("pics/taskF_battery_vs_ocgt.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: pics/taskF_battery_vs_ocgt.png")


# =============================================================================
# STEP 5 — Key thresholds for report discussion
# =============================================================================
print("\n" + "=" * 60)
print("KEY THRESHOLDS FOR REPORT DISCUSSION")
print("=" * 60)

# At what budget does OCGT capacity drop below 0.1 GW?
ocgt_small = df[df["OCGT_GW"] < 0.1]
if not ocgt_small.empty:
    row = ocgt_small.iloc[0]
    print(f"OCGT < 0.1 GW at CO2 budget : {row['co2_limit_Mt']:.2f} MtCO2/year "
          f"({row['pct_free']:.0f}% of unconstrained)")

# At what budget does shadow price enter EU ETS range (>= 60 EUR/t)?
ets_enter = df[df["co2_price_EUR"] >= 60]
if not ets_enter.empty:
    row = ets_enter.iloc[-1]   # last point where it's still >= 60 (loosest budget)
    print(f"Shadow price enters ETS range at : {row['co2_limit_Mt']:.2f} MtCO2/year")

# Cost increase going from unconstrained to near-zero
cost_free  = df.iloc[-1]["cost_MEur"]   # unconstrained (last row, highest limit)
cost_tight = df.iloc[0]["cost_MEur"]    # tightest constraint (first row)
print(f"Cost at unconstrained         : {cost_free:,.0f} M€/year")
print(f"Cost at tightest constraint   : {cost_tight:,.0f} M€/year")
print(f"Cost increase (tight vs free) : {(cost_tight/cost_free - 1)*100:.1f}%")

print("\n" + "=" * 60)
print("FULL RESULTS TABLE")
print("=" * 60)
print(df[["co2_limit_Mt", "Wind_GW", "Solar_GW", "OCGT_GW",
          "Battery_GW", "cost_MEur", "co2_price_EUR"]].to_string(
    index=False, float_format=lambda v: f"{v:.2f}"))
