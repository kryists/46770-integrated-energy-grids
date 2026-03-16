import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypsa

def annuity(y, r):
    if r > 0:
      return r / (1.0 - 1.0 / (1.0 + r) ** y)
    else:
       return 1/y
    

def build_and_solve(year: int) -> pypsa.Network:
     
    n = pypsa.Network()
    snapshots = pd.date_range(f'{year}-01-01 00:00Z', f'{year}-12-31 23:00Z', freq='h')
    n.set_snapshots(snapshots.values)
    n.add("Bus", "ITA")
 
    # Demand Data
    demand = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)
    demand.index = pd.to_datetime(demand.index)
    country = "ITA"
    n.add('Load', "Demand", bus='ITA', p_set=demand[country].values)
 
    # Carriers
    n.add("Carrier", "Gas", co2_emissions=0.19)
    n.add("Carrier", "Wind")
    n.add("Carrier", "Solar")
 
    # Wind Data
    data_wind = pd.read_csv('data/onshore_wind_1979-2017.csv', sep=';', index_col=0)
    data_wind.index = pd.to_datetime(data_wind.index, utc=True)
    cf_wind = data_wind[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ") for h in n.snapshots]]
    capital_cost_wind = annuity(30, 0.07) * 910_000 * (1 + 0.033)
    n.add("Generator", "Wind",
          bus="ITA", p_nom_extendable=True, carrier="Wind",
          capital_cost=capital_cost_wind, marginal_cost=0,
          p_max_pu=cf_wind.values)
 
    # Solar Data
    data_solar = pd.read_csv('data/pv_optimal.csv', sep=';', index_col=0)
    data_solar.index = pd.to_datetime(data_solar.index)
    cf_solar = data_solar[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ") for h in n.snapshots]]
    capital_cost_solar = annuity(25, 0.07) * 425_000 * (1 + 0.03)
    n.add("Generator", "Solar",
          bus="ITA", p_nom_extendable=True, carrier="Solar",
          capital_cost=capital_cost_solar, marginal_cost=0,
          p_max_pu=cf_solar.values)
 
    # Gas Data
    capital_cost_OCGT = annuity(25, 0.07) * 560_000 * (1 + 0.033)
    fuel_cost  = 21.6       # €/MWh_th
    efficiency = 0.39
    n.add("Generator", "OCGT",
          bus="ITA", carrier="Gas",
          capital_cost=capital_cost_OCGT,
          marginal_cost=fuel_cost / efficiency,
          p_nom_extendable=True)
    

 
    n.optimize(solver_name='gurobi')
    return n


print('=' * 50)
print('Part A')
print('=' * 50)
print("\nSolving 2010 model …\n")
n = build_and_solve(2010)
 
print(f"Total cost : {n.objective / 1e6:.2f} million €")
print(f"Cost/MWh   : {n.objective / n.loads_t.p.sum().values[0]:.2f} €/MWh")
print("Optimal capacities [MW]:")
print(n.generators.p_nom_opt)
 
COLORS = {'Wind': 'blue', 'Solar': 'orange', 'OCGT': 'grey', 'Demand': 'black'}
 
# Dispatch – summer week (first week of July)
summer_start = pd.Timestamp('2010-07-06')
summer_slice  = slice(summer_start, summer_start + pd.Timedelta(hours=167))
 
fig, ax = plt.subplots(figsize=(12, 4))
ts = n.generators_t.p.loc[summer_slice]
dem = n.loads_t.p['Demand'].loc[summer_slice]
ax.stackplot(ts.index, ts['Wind'], ts['Solar'], ts['OCGT'],
             labels=['Wind', 'Solar', 'OCGT'],
             colors=[COLORS['Wind'], COLORS['Solar'], COLORS['OCGT']], alpha=0.85)
ax.plot(dem.index, dem.values, color='black', lw=1.5, label='Demand')
ax.set_title('Dispatch - Summer Week (July 2010)')
ax.set_ylabel('Power [MW]')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_xlabel('Date')
fig.tight_layout()
#plt.savefig('dispatch_summer.png', dpi=150)
plt.show()
 
# Dispatch – winter week (first week of January)
winter_start = pd.Timestamp('2010-01-05')
winter_slice  = slice(winter_start, winter_start + pd.Timedelta(hours=167))
 
fig, ax = plt.subplots(figsize=(12, 4))
ts = n.generators_t.p.loc[winter_slice]
dem = n.loads_t.p['Demand'].loc[winter_slice]
ax.stackplot(ts.index, ts['Wind'], ts['Solar'], ts['OCGT'],
             labels=['Wind', 'Solar', 'OCGT'],
             colors=[COLORS['Wind'], COLORS['Solar'], COLORS['OCGT']], alpha=0.85)
ax.plot(dem.index, dem.values, color='black', lw=1.5, label='Demand')
ax.set_title('Dispatch - Winter Week (January 2010)')
ax.set_ylabel('Power [MW]')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_xlabel('Date')
fig.tight_layout()
#plt.savefig('dispatch_winter.png', dpi=150)
plt.show()
 
# Annual electricity mix (pie)
mix = {g: n.generators_t.p[g].sum() for g in ['Wind', 'Solar', 'OCGT']}
fig, ax = plt.subplots(figsize=(5, 5))
wedges, texts, autotexts = ax.pie(
    mix.values(),
    labels=mix.keys(),
    colors=[COLORS[g] for g in mix],
    autopct='%1.1f%%',
    wedgeprops={'linewidth': 0.8, 'edgecolor': 'white'},
    startangle=140,
)
for at in autotexts:
    at.set_fontsize(9)
ax.set_title('Annual electricity mix - Italy 2010', y=1.03)
fig.tight_layout()
#plt.savefig('electricity_mix.png', dpi=150)
plt.show()
 
# Duration curves
fig, ax = plt.subplots(figsize=(10, 5))
hours = np.arange(1, 8761)
for gen, col in [('Wind', COLORS['Wind']),
                 ('Solar', COLORS['Solar']),
                 ('OCGT', COLORS['OCGT'])]:
    sorted_out = np.sort(n.generators_t.p[gen].values)[::-1]
    ax.plot(hours, sorted_out, color=col, lw=1.5, label=gen)
 
# Demand duration curve for reference
sorted_dem = np.sort(n.loads_t.p['Demand'].values)[::-1]
ax.plot(hours, sorted_dem, color='black', lw=1.2, ls='--', label='Demand')
 
ax.set_title('Duration curves - Italy 2010')
ax.set_xlabel('Hours (sorted)')
ax.set_ylabel('Power [MW]')
ax.legend(framealpha=0.9)
ax.set_xlim(0, 8760)
ax.set_ylim(bottom=0)
fig.tight_layout()
#plt.savefig('duration_curves.png', dpi=150)
plt.show()
 
# Capacity factors
# CF = mean(p_t) / p_nom_opt
cf_bars = {}
for gen in ['Wind', 'Solar', 'OCGT']:
    p_nom = n.generators.loc[gen, 'p_nom_opt']
    cf_bars[gen] = n.generators_t.p[gen].mean() / p_nom if p_nom > 0 else 0.0
 
fig, ax = plt.subplots(figsize=(5, 4))
gens = list(cf_bars.keys())
vals = [cf_bars[g] for g in gens]
bars = ax.bar(gens, vals,
              color=[COLORS[g] for g in gens],
              edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + 0.01, f'{val:.2f}',
            ha='center', va='bottom', fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_title('Capacity factors - Italy 2010')
ax.set_ylabel('Capacity factor [ ]')
ax.set_xlabel('Generator')
fig.tight_layout()
#plt.savefig('capacity_factors.png', dpi=150)
plt.show()

### Part B
print("=" * 50)
print("Part B")
print('=' * 50)
YEARS = [1990, 1991, 1993, 1994, 1995]
results = {g: [] for g in ['Wind', 'Solar', 'OCGT']}  # optimal capacities
cf_results = {g: [] for g in ['Wind', 'Solar', 'OCGT']}  # capacity factors
 
for yr in YEARS:
    print(f"\nSolving {yr} model …")
    try:
        nn = build_and_solve(yr)
        for gen in ['Wind', 'Solar', 'OCGT']:
            p_nom = nn.generators.loc[gen, 'p_nom_opt']
            results[gen].append(p_nom)
            cf = nn.generators_t.p[gen].mean() / p_nom if p_nom > 0 else 0.0
            cf_results[gen].append(cf)
        print(f"  {yr}: Wind={results['Wind'][-1]:.0f} MW, "
              f"Solar={results['Solar'][-1]:.0f} MW, "
              f"OCGT={results['OCGT'][-1]:.0f} MW")
    except Exception as e:
        print(f"  {yr} failed: {e}")
        for gen in ['Wind', 'Solar', 'OCGT']:
            results[gen].append(np.nan)
            cf_results[gen].append(np.nan)
 
# Optimal capacity across weather years
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
# Grouped bar chart: capacity
x     = np.arange(len(YEARS))
width = 0.25
ax = axes[0]
for i, (gen, col) in enumerate([('Wind',  COLORS['Wind']),
                                 ('Solar', COLORS['Solar']),
                                 ('OCGT',  COLORS['OCGT'])]):
    ax.bar(x + i * width, results[gen], width,
           label=gen, color=col, edgecolor='white', linewidth=0.6)
ax.set_xticks(x + width)
ax.set_xticklabels(YEARS)
ax.set_ylabel('Optimal capacity [MW]')
ax.set_title('Optimal capacity per weather year')
ax.legend()
 
# Add mean ± std annotation 
for i, (gen, col) in enumerate([('Wind',  COLORS['Wind']),
                                 ('Solar', COLORS['Solar']),
                                 ('OCGT',  COLORS['OCGT'])]):
    vals = np.array(results[gen], dtype=float)
    mn, sd = np.nanmean(vals), np.nanstd(vals)
    ax.axhline(mn, color=col, ls='--', lw=0.8, alpha=0.5)
 
# Mean ± std summary bar chart
ax2 = axes[1]
gens  = ['Wind', 'Solar', 'OCGT']
means = [np.nanmean(results[g]) for g in gens]
stds  = [np.nanstd(results[g])  for g in gens]
bars2 = ax2.bar(gens, means, color=[COLORS[g] for g in gens],
                edgecolor='white', linewidth=0.8, alpha=0.85)
ax2.errorbar(gens, means, yerr=stds, fmt='none',
             ecolor='black', capsize=6, elinewidth=1.2)
for bar, mn, sd in zip(bars2, means, stds):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             mn + sd + max(means) * 0.01,
             f'{mn:.0f}\n±{sd:.0f}', ha='center', va='bottom', fontsize=8)
ax2.set_ylabel('Capacity [MW]')
ax2.set_title('Mean ± std of optimal capacity\n(2013-2017)')
 
fig.tight_layout()
#plt.savefig('interannual_capacity.png', dpi=150)
plt.show()
 
# Capacity factor variability across weather years
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
ax = axes[0]
for gen, col in [('Wind',  COLORS['Wind']),
                 ('Solar', COLORS['Solar']),
                 ('OCGT',  COLORS['OCGT'])]:
    ax.plot(YEARS, cf_results[gen], 'o-', color=col, label=gen, lw=1.5, ms=6)
ax.set_title('Capacity factor per weather year')
ax.set_ylabel('Capacity factor [ ]')
ax.set_xlabel('Year')
ax.set_ylim(0, 1)
ax.legend()
ax.set_xticks(YEARS)
 
ax2 = axes[1]
cf_means = [np.nanmean(cf_results[g]) for g in gens]
cf_stds  = [np.nanstd(cf_results[g])  for g in gens]
bars3 = ax2.bar(gens, cf_means, color=[COLORS[g] for g in gens],
                edgecolor='white', linewidth=0.8, alpha=0.85)
ax2.errorbar(gens, cf_means, yerr=cf_stds, fmt='none',
             ecolor='black', capsize=6, elinewidth=1.2)
for bar, mn, sd in zip(bars3, cf_means, cf_stds):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             mn + sd + 0.01,
             f'{mn:.2f}\n±{sd:.2f}', ha='center', va='bottom', fontsize=8)
ax2.set_ylabel('Capacity factor [ ]')
ax2.set_ylim(0, 1)
ax2.set_title('Mean ± std of capacity factor\n(2013-2017)')
 
fig.tight_layout()
#plt.savefig('interannual_cf.png', dpi=150)
plt.show()

print("DONE!!!!")