import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('household_power_consumption.txt', 
                  sep=';', 
                  low_memory=False,
                  na_values=['?'])

# Combine date and time into one column
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                 format='%d/%m/%Y %H:%M:%S')

# Convert power column to numbers
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

# Drop missing values
df = df.dropna(subset=['Global_active_power'])

# Plot one week of data
one_week = df[df['datetime'].dt.date.astype(str) < '2007-01-15']
plt.figure(figsize=(12, 4))
plt.plot(one_week['datetime'], one_week['Global_active_power'], color='steelblue', linewidth=0.8)
plt.title('Household Energy Consumption - One Week')
plt.xlabel('Time')
plt.ylabel('Power (kW)')
plt.tight_layout()
plt.show()

print(f"Dataset loaded: {len(df)} rows")
print(df[['datetime', 'Global_active_power']].head(10))