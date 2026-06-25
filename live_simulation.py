import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import time

# ── Time-of-use tariff ────────────────────────────────────────────────────────
def get_tariff(hour, day_of_week):
    """
    Time-of-use tariff structure.
    day_of_week: 0=Monday, 6=Sunday
    Returns price in €/kWh
    """
    is_weekend = day_of_week >= 5
    if is_weekend:
        return 0.12                    # weekend flat rate
    elif 7 <= hour < 22:
        return 0.20                    # weekday peak
    else:
        return 0.10                    # weekday off-peak

def get_tariff_period(hour, day_of_week):
    if day_of_week >= 5:
        return 'Weekend'
    elif 7 <= hour < 22:
        return 'Peak'
    else:
        return 'Off-Peak'

# ── 1. Connect to Firebase ────────────────────────────────────────────────────
cred = credentials.Certificate('firebase_key.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# ── 2. Load UCI Dataset ───────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('household_power_consumption.txt',
                 sep=';', low_memory=False, na_values=['?'])

df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                format='%d/%m/%Y %H:%M:%S')
df['Global_active_power'] = pd.to_numeric(
    df['Global_active_power'], errors='coerce')
df = df.dropna(subset=['Global_active_power'])
df = df.set_index('datetime')

week = df['2007-01-01':'2007-01-07']['Global_active_power'].copy()
print(f"Dataset ready: {len(week)} readings\n")

# ── 3. Default appliance states ───────────────────────────────────────────────
appliance_states = {
    'fridge':          True,
    'water_heater':    True,
    'washing_machine': True,
    'oven':            True,
    'tv':              True,
    'lighting':        True,
}

# ── 4. Read device commands from Firebase ─────────────────────────────────────
def read_device_commands():
    try:
        doc = db.collection('device_commands').document('current').get()
        if doc.exists:
            data = doc.to_dict()
            for key in appliance_states:
                if key in data:
                    appliance_states[key] = bool(data[key])
            print(f"  Commands read: "
                  f"{ {k: ('ON' if v else 'OFF') for k,v in appliance_states.items()} }")
    except Exception as e:
        print(f"  Could not read commands: {e}")

# ── 5. Appliance simulation ───────────────────────────────────────────────────
def get_appliance_breakdown(total_power, hour):
    fridge          = 0.15 if appliance_states['fridge']          else 0.0
    water_heater    = 3.0  if appliance_states['water_heater']    else 0.0
    washing_machine = 2.0  if appliance_states['washing_machine'] else 0.0
    oven            = 2.0  if appliance_states['oven']            else 0.0
    tv              = 0.12 if appliance_states['tv']              else 0.0
    lighting        = 0.06 if appliance_states['lighting']        else 0.0

    # Small random variation
    fridge       += np.random.uniform(-0.02, 0.02) if fridge > 0       else 0
    water_heater += np.random.uniform(-0.2,  0.2)  if water_heater > 0 else 0

    simulated_total = max(0, fridge + washing_machine + tv +
                          lighting + water_heater + oven)

    print(f"  DEBUG: fridge={fridge:.2f} wat={water_heater:.2f} "
          f"was={washing_machine:.2f} oven={oven:.2f} "
          f"tv={tv:.2f} lig={lighting:.2f} "
          f"total={simulated_total:.2f}")

    if simulated_total > 0:
        blended = (total_power * 0.2) + (simulated_total * 0.8)
    else:
        blended = 0.15 + np.random.uniform(0, 0.05)

    return {
        'fridge':          round(max(0, fridge), 3),
        'washing_machine': round(max(0, washing_machine), 3),
        'tv':              round(max(0, tv), 3),
        'lighting':        round(max(0, lighting), 3),
        'water_heater':    round(max(0, water_heater), 3),
        'oven':            round(max(0, oven), 3),
        'blended_total':   round(blended, 3),
    }

# ── 6. Stream data to Firebase ────────────────────────────────────────────────
print("Starting live simulation...")
print("Tariff: Peak €0.20/kWh (07-22 weekdays) | "
      "Off-peak €0.10/kWh (22-07 weekdays) | "
      "Weekend €0.12/kWh")
print("Press Ctrl+C to stop\n")

minutes_to_hours = 1 / 60
running_kwh      = 0.0
running_cost     = 0.0
reading_count    = 0

for timestamp, total_power in week.items():
    # Read device commands every 5 readings
    if reading_count % 5 == 0:
        read_device_commands()

    now        = datetime.now()
    hour       = now.hour
    dow        = now.weekday()
    tariff     = get_tariff(hour, dow)
    period     = get_tariff_period(hour, dow)
    appliances = get_appliance_breakdown(float(total_power), hour)

    # Accumulate energy and cost using TOU tariff
    kwh_this_reading  = appliances['blended_total'] * minutes_to_hours
    cost_this_reading = kwh_this_reading * tariff
    running_kwh      += kwh_this_reading
    running_cost     += cost_this_reading

    reading = {
        'timestamp':             now.isoformat(),
        'simulated_time':        timestamp.isoformat(),
        'total_power':           appliances['blended_total'],
        'fridge':                appliances['fridge'],
        'washing_machine':       appliances['washing_machine'],
        'tv':                    appliances['tv'],
        'lighting':              appliances['lighting'],
        'water_heater':          appliances['water_heater'],
        'oven':                  appliances['oven'],
        'running_kwh':           round(running_kwh, 3),
        'estimated_bill_so_far': round(running_cost, 2),
        'current_tariff':        tariff,
        'tariff_period':         period,
        'appliance_states':      {k: v for k, v in appliance_states.items()},
    }

    db.collection('energy_readings').add(reading)

    status = ' | '.join([
        f"{k[:3].upper()}:{'ON' if v else 'OFF'}"
        for k, v in appliance_states.items()
    ])
    print(f"[{now.strftime('%H:%M')}] "
          f"Power: {appliances['blended_total']:.2f} kW | "
          f"Tariff: €{tariff:.2f}/kWh ({period}) | "
          f"Bill: €{running_cost:.2f} | "
          f"{status}")

    reading_count += 1
    time.sleep(30)

print("\nDone!")