import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Load the UCI dataset ──────────────────────────────────────────────────
df = pd.read_csv('household_power_consumption.txt',
                 sep=';',
                 low_memory=False,
                 na_values=['?'])

df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                format='%d/%m/%Y %H:%M:%S')
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
df = df.dropna(subset=['Global_active_power'])
df = df.set_index('datetime')

# ── 2. Take one week of data to work with ────────────────────────────────────
week = df['2007-01-01':'2007-01-07']['Global_active_power'].copy()

# ── 3. Appliance simulation ───────────────────────────────────────────────────
# For each minute, we estimate how much each appliance contributes
# based on time of day and typical wattage (in kW)

def simulate_appliances(datetime_index):
    n = len(datetime_index)
    hour = datetime_index.hour

    # Fridge: always on, cycles every 15 min (0.15 kW average)
    fridge = np.full(n, 0.15)

    # Washing machine: mornings 8-10, evenings 18-20 (2.0 kW when on)
    washing = np.where(
        ((hour >= 8) & (hour < 10)) | ((hour >= 18) & (hour < 20)),
        2.0, 0.0
    )

    # TV: evenings 19-23 (0.1 kW)
    tv = np.where((hour >= 19) & (hour < 23), 0.1, 0.0)

    # Lighting: evenings and early morning 6-8 and 18-23 (0.06 kW)
    lighting = np.where(
        ((hour >= 6) & (hour < 8)) | ((hour >= 18) & (hour < 23)),
        0.06, 0.0
    )

    # Water heater: mornings 6-9, evenings 18-21 (3.0 kW when on)
    water_heater = np.where(
        ((hour >= 6) & (hour < 9)) | ((hour >= 18) & (hour < 21)),
        3.0, 0.0
    )

    # Oven: lunch 12-13, dinner 18-20 (2.0 kW when on)
    oven = np.where(
        ((hour >= 12) & (hour < 13)) | ((hour >= 18) & (hour < 20)),
        2.0, 0.0
    )

    return pd.DataFrame({
        'Fridge':        fridge,
        'Washing Machine': washing,
        'TV':            tv,
        'Lighting':      lighting,
        'Water Heater':  water_heater,
        'Oven':          oven,
    }, index=datetime_index)

appliances = simulate_appliances(week.index)

# ── 4. Calculate simulated total and bill ────────────────────────────────────
appliances['Simulated Total'] = appliances.sum(axis=1)
appliances['Real Total']      = week.values

# Bill calculation (kWh per minute → sum → multiply by price)
# 1 minute = 1/60 hour, price = 0.15 EUR per kWh (adjust to your country)
price_per_kwh = 0.15
minutes_to_hours = 1 / 60

weekly_kwh = appliances['Simulated Total'].sum() * minutes_to_hours
weekly_bill = weekly_kwh * price_per_kwh

print("=" * 45)
print("  APPLIANCE ENERGY SUMMARY — ONE WEEK")
print("=" * 45)
for appliance in ['Fridge','Washing Machine','TV','Lighting','Water Heater','Oven']:
    kwh   = appliances[appliance].sum() * minutes_to_hours
    cost  = kwh * price_per_kwh
    print(f"  {appliance:<20} {kwh:>7.2f} kWh   €{cost:.2f}")
print("-" * 45)
print(f"  {'TOTAL':<20} {weekly_kwh:>7.2f} kWh   €{weekly_bill:.2f}")
print("=" * 45)

# ── 5. Plot appliance breakdown ───────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Chart 1 - Real vs Simulated total
axes[0].plot(appliances.index, appliances['Real Total'],
             label='Real Total', color='steelblue', linewidth=0.6, alpha=0.8)
axes[0].plot(appliances.index, appliances['Simulated Total'],
             label='Simulated Total', color='orange', linewidth=0.8, alpha=0.9)
axes[0].set_title('Real vs Simulated Total Power')
axes[0].set_ylabel('Power (kW)')
axes[0].legend()

# Chart 2 - Appliance breakdown stacked
appliance_cols = ['Fridge','Washing Machine','TV','Lighting','Water Heater','Oven']
axes[1].stackplot(appliances.index,
                  [appliances[col] for col in appliance_cols],
                  labels=appliance_cols,
                  alpha=0.8)
axes[1].set_title('Appliance Breakdown (Simulated)')
axes[1].set_ylabel('Power (kW)')
axes[1].legend(loc='upper right', fontsize=8)

# Chart 3 - Daily bill estimate
appliances['date'] = appliances.index.date
daily_kwh  = appliances.groupby('date')['Simulated Total'].sum() * minutes_to_hours
daily_bill = daily_kwh * price_per_kwh
axes[2].bar(range(len(daily_bill)), daily_bill.values, color='steelblue', alpha=0.8)
axes[2].set_xticks(range(len(daily_bill)))
axes[2].set_xticklabels([str(d) for d in daily_bill.index], rotation=30, fontsize=8)
axes[2].set_title('Estimated Daily Bill (EUR)')
axes[2].set_ylabel('Cost (€)')

plt.tight_layout()
plt.show()