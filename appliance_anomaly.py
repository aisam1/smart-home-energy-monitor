import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import firebase_admin
from firebase_admin import credentials, firestore
import warnings
warnings.filterwarnings('ignore')

# ── 1. Connect to Firebase ────────────────────────────────────────────────────
if not firebase_admin._apps:
    cred = credentials.Certificate('firebase_key.json')
    firebase_admin.initialize_app(cred)
db = firestore.client()

# ── 2. Load UCI dataset ───────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('household_power_consumption.txt',
                 sep=';', low_memory=False, na_values=['?'])

df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                format='%d/%m/%Y %H:%M:%S')

cols = ['Global_active_power', 'Sub_metering_1',
        'Sub_metering_2', 'Sub_metering_3']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=cols)
df['hour']        = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek

df['total_power']     = df['Global_active_power']
df['kitchen']         = df['Sub_metering_1']  # keep in Wh, not kWh
df['laundry']         = df['Sub_metering_2']
df['water_heater_ac'] = df['Sub_metering_3']
df['other']           = (df['Global_active_power'] * (1000/60)) - \
                         df['Sub_metering_1'] - \
                         df['Sub_metering_2'] - \
                         df['Sub_metering_3']
df['other'] = df['other'].clip(lower=0)

print(f"Loaded {len(df)} readings")

# ── 3. Appliance anomaly detection ────────────────────────────────────────────
appliances = {
    'Kitchen (Oven/Dishwasher)': {'col': 'kitchen',         'unit': 'Wh', 'min_val': 10},
    'Laundry (Washing Machine)': {'col': 'laundry',         'unit': 'Wh', 'min_val': 10},
    'Water Heater & AC':         {'col': 'water_heater_ac', 'unit': 'Wh', 'min_val': 10},
    'Other (Lighting/TV)':       {'col': 'other',           'unit': 'Wh', 'min_val': 50},
    'Total Power':               {'col': 'total_power',     'unit': 'kW', 'min_val': 1},
}

all_alerts = []

print("\n" + "=" * 55)
print("  APPLIANCE-LEVEL ANOMALY DETECTION")
print("=" * 55)

for appliance_name, config in appliances.items():
    col     = config['col']
    unit    = config['unit']
    min_val = config['min_val']

    # Hourly baseline
    hourly_avg = df.groupby('hour')[col].mean()
    hourly_std = df.groupby('hour')[col].std().clip(lower=0.1)

    df[f'{col}_expected'] = df['hour'].map(hourly_avg)
    df[f'{col}_std']      = df['hour'].map(hourly_std)
    df[f'{col}_zscore']   = (df[col] - df[f'{col}_expected']) / \
                             df[f'{col}_std']

    # Run Isolation Forest
    features = df[[col, f'{col}_zscore', 'hour']].values
    scaler   = StandardScaler()
    scaled   = scaler.fit_transform(features)

    iso = IsolationForest(contamination=0.02, random_state=42)
    df[f'{col}_anomaly'] = iso.fit_predict(scaled)

    # Get anomalies above minimum threshold
    anomalies = df[
        (df[f'{col}_anomaly'] == -1) &
        (df[col] >= min_val)
    ].copy()

    if len(anomalies) == 0:
        print(f"\n{appliance_name}: No significant anomalies found")
        continue

    # Use z-score for severity
    anomalies['severity_score'] = anomalies[f'{col}_zscore'].abs()

    # Classify severity by z-score
    anomalies['severity'] = pd.cut(
        anomalies['severity_score'],
        bins=[0, 2.5, 4, float('inf')],
        labels=['warning', 'high', 'critical']
    )

    print(f"\n{appliance_name}:")
    print(f"  Anomalies detected: {len(anomalies)}")
    print(f"  Warning:  {sum(anomalies['severity'] == 'warning')}")
    print(f"  High:     {sum(anomalies['severity'] == 'high')}")
    print(f"  Critical: {sum(anomalies['severity'] == 'critical')}")

    # Top 4 per appliance
    top_anomalies = anomalies.nlargest(4, 'severity_score')

    for _, row in top_anomalies.iterrows():
        severity = str(row['severity'])
        zscore   = row['severity_score']
        actual   = row[col]
        expected = row[f'{col}_expected']

        if actual > expected:
            direction = f"{actual:.1f} {unit} vs expected {expected:.1f} {unit}"
            action    = "above"
        else:
            direction = f"{actual:.1f} {unit} vs expected {expected:.1f} {unit}"
            action    = "below"

        hour = int(row['hour'])
        if 0 <= hour < 6:
            time_context = "late night"
        elif 6 <= hour < 12:
            time_context = "morning"
        elif 12 <= hour < 18:
            time_context = "afternoon"
        else:
            time_context = "evening"

        message = (f"{appliance_name} is {action} normal at "
                   f"{time_context} — {direction} "
                   f"(z-score: {zscore:.1f})")

        alert = {
            'timestamp':      row['datetime'].isoformat(),
            'appliance':      appliance_name,
            'severity':       severity,
            'actual_value':   round(float(actual), 2),
            'expected_value': round(float(expected), 2),
            'zscore':         round(float(zscore), 2),
            'unit':           unit,
            'hour':           hour,
            'message':        message,
        }
        all_alerts.append(alert)

# ── 4. Balance and sort alerts ────────────────────────────────────────────────
severity_order = {'critical': 0, 'high': 1, 'warning': 2}

alerts_by_appliance = defaultdict(list)
for alert in all_alerts:
    alerts_by_appliance[alert['appliance']].append(alert)

balanced_alerts = []
for appliance, alerts in alerts_by_appliance.items():
    alerts.sort(key=lambda x: -x['zscore'])
    balanced_alerts.extend(alerts[:4])

balanced_alerts.sort(key=lambda x: severity_order.get(x['severity'], 3))
all_alerts = balanced_alerts

print(f"\n{'=' * 55}")
print(f"  TOTAL ALERTS: {len(all_alerts)}")
print(f"  Critical: {sum(1 for a in all_alerts if a['severity'] == 'critical')}")
print(f"  High:     {sum(1 for a in all_alerts if a['severity'] == 'high')}")
print(f"  Warning:  {sum(1 for a in all_alerts if a['severity'] == 'warning')}")
print(f"{'=' * 55}")

print("\nOne sample alert per appliance:")
seen = set()
for alert in all_alerts:
    if alert['appliance'] not in seen:
        seen.add(alert['appliance'])
        print(f"  [{alert['severity'].upper():8}] "
              f"{alert['appliance']}: {alert['message']}")

# ── 5. Save to Firebase ───────────────────────────────────────────────────────
print("\nSaving alerts to Firebase...")

old_docs = db.collection('appliance_alerts').limit(500).stream()
for doc in old_docs:
    doc.reference.delete()

batch_size = 50
for i in range(0, min(len(all_alerts), 500), batch_size):
    batch = db.batch()
    for alert in all_alerts[i:i + batch_size]:
        ref = db.collection('appliance_alerts').document()
        batch.set(ref, alert)
    batch.commit()
    print(f"  Saved alerts {i+1} to {min(i+batch_size, len(all_alerts))}")

print(f"\nDone! {min(len(all_alerts), 500)} alerts saved to Firebase")
print("Collection: appliance_alerts")