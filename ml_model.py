import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import firebase_admin
from firebase_admin import credentials, firestore

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

# ── 1. Load UCI dataset ───────────────────────────────────────────────────────
print("Loading UCI dataset...")
df = pd.read_csv('household_power_consumption.txt',
                 sep=';',
                 low_memory=False,
                 na_values=['?'])

df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                format='%d/%m/%Y %H:%M:%S')
df['Global_active_power'] = pd.to_numeric(
    df['Global_active_power'], errors='coerce')
df = df.dropna(subset=['Global_active_power'])
df = df.sort_values('datetime').reset_index(drop=True)
df['total_power'] = df['Global_active_power']

# Feature engineering
df['hour']        = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month']       = df['datetime'].dt.month
df['date']        = df['datetime'].dt.date

# Apply TOU tariff to every reading
df['tariff']      = df.apply(
    lambda row: get_tariff(row['hour'], row['day_of_week']),
    axis=1)
df['tariff_period'] = df.apply(
    lambda row: get_tariff_period(row['hour'], row['day_of_week']),
    axis=1)
df['kwh']         = df['total_power'] * (1/60)
df['cost']        = df['kwh'] * df['tariff']

print(f"Loaded {len(df)} minute readings")

# ── 2. ANOMALY DETECTION ──────────────────────────────────────────────────────
print("=" * 50)
print("  ANOMALY DETECTION")
print("=" * 50)

hourly_avg            = df.groupby('hour')['total_power'].mean()
df['expected_power']  = df['hour'].map(hourly_avg)
df['power_deviation'] = df['total_power'] - df['expected_power']

features_anomaly = df[['total_power', 'power_deviation', 'hour']].copy()
scaler           = StandardScaler()
features_scaled  = scaler.fit_transform(features_anomaly)

iso_forest       = IsolationForest(contamination=0.03, random_state=42)
df['anomaly']    = iso_forest.fit_predict(features_scaled)
df['anomaly_label'] = df['anomaly'].map({1: 'Normal', -1: '⚠️ Anomaly'})

anomalies = df[df['anomaly'] == -1]
print(f"Total readings analyzed: {len(df)}")
print(f"Anomalies detected: {len(anomalies)} "
      f"({len(anomalies)/len(df)*100:.1f}%)\n")
print("Sample anomalous readings (genuine spikes):")
top_anomalies = anomalies.nlargest(5, 'power_deviation')
print(top_anomalies[['datetime', 'total_power',
                      'expected_power', 'power_deviation',
                      'anomaly_label']].to_string())

# ── Isolation Forest evaluation plots ────────────────────────────────────────
print("\nGenerating Isolation Forest evaluation plots...")

sample     = df[df['datetime'].dt.date.astype(str)
                .between('2007-01-01', '2007-01-07')].copy()
normal_s   = sample[sample['anomaly'] ==  1]
anomaly_s  = sample[sample['anomaly'] == -1]

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(normal_s['datetime'], normal_s['total_power'],
             color='steelblue', linewidth=0.8,
             label='Normal', alpha=0.8)
axes[0].scatter(anomaly_s['datetime'], anomaly_s['total_power'],
                color='red', s=20, zorder=5,
                label=f'Anomaly ({len(anomaly_s)} points)')
axes[0].set_title('Isolation Forest: Anomaly Detection — Jan 2007')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Power (kW)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

features_plot = df[['total_power', 'power_deviation', 'hour']].copy()
scaler_plot   = StandardScaler()
features_sc   = scaler_plot.fit_transform(features_plot)
iso_scores    = IsolationForest(contamination=0.03, random_state=42)
iso_scores.fit(features_sc)
scores        = iso_scores.score_samples(features_sc)

axes[1].hist(scores[df['anomaly'] ==  1], bins=80,
             alpha=0.6, color='steelblue',
             label='Normal readings', density=True)
axes[1].hist(scores[df['anomaly'] == -1], bins=80,
             alpha=0.6, color='red',
             label='Anomalies', density=True)
axes[1].set_title('Isolation Forest: Anomaly Score Distribution')
axes[1].set_xlabel('Anomaly Score (lower = more anomalous)')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('isolation_forest_evaluation.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: isolation_forest_evaluation.png")

print(f"\nIsolation Forest Summary:")
print(f"  Total readings:     {len(df):,}")
print(f"  Anomalies detected: {len(df[df['anomaly']==-1]):,} "
      f"({len(df[df['anomaly']==-1])/len(df)*100:.1f}%)")
print(f"  Contamination set:  3.0%")
print(f"  Top anomaly hours:  "
      f"{df[df['anomaly']==-1].groupby('hour').size().nlargest(3).index.tolist()}")
print(f"  Avg anomaly power:  "
      f"{df[df['anomaly']==-1]['total_power'].mean():.3f} kW")
print(f"  Avg normal power:   "
      f"{df[df['anomaly']==1]['total_power'].mean():.3f} kW")

# ── 3. BILL PREDICTION ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  DAILY BILL PREDICTION (TOU tariff)")
print("=" * 50)

# Daily bill using TOU tariff — sum of tariff-adjusted costs per day
daily = df.groupby('date').agg(
    daily_kwh=('kwh', 'sum'),
    daily_bill=('cost', 'sum')    # TOU-adjusted cost
).reset_index()

daily['date']        = pd.to_datetime(daily['date'])
daily['day_of_week'] = daily['date'].dt.dayofweek
daily['month']       = daily['date'].dt.month
daily['day_num']     = range(len(daily))

print(f"Daily bill range: "
      f"€{daily['daily_bill'].min():.2f} — "
      f"€{daily['daily_bill'].max():.2f}")
print(f"Average daily bill (TOU): "
      f"€{daily['daily_bill'].mean():.2f}")

# Lag features
daily['prev_day_kwh']    = daily['daily_kwh'].shift(1)
daily['prev_7_day_kwh']  = daily['daily_kwh'].rolling(7).mean()
daily['prev_14_day_kwh'] = daily['daily_kwh'].rolling(14).mean()
daily['prev_30_day_kwh'] = daily['daily_kwh'].rolling(30).mean()
daily = daily.dropna()

features = daily[['day_of_week', 'month', 'prev_day_kwh',
                   'prev_7_day_kwh', 'prev_14_day_kwh',
                   'prev_30_day_kwh']]
target   = daily['daily_bill']

# Chronological split
split   = int(len(features) * 0.8)
X_train = features.iloc[:split]
X_test  = features.iloc[split:]
y_train = target.iloc[:split]
y_test  = target.iloc[split:]

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred      = rf_model.predict(X_test)
train_score = rf_model.score(X_train, y_train)
test_score  = rf_model.score(X_test,  y_test)
mae         = mean_absolute_error(y_test, y_pred)
rmse        = np.sqrt(np.mean((y_test - y_pred) ** 2))

print(f"\nModel Performance (TOU tariff):")
print(f"  Training R²: {train_score:.3f}")
print(f"  Test R²:     {test_score:.3f}")
print(f"  MAE:         €{mae:.3f}/day")
print(f"  RMSE:        €{rmse:.3f}/day\n")

# ── Predict next 7 days from today ────────────────────────────────────────────
from datetime import datetime
today       = datetime.now()
today_dow   = today.weekday()
today_month = today.month

avg_kwh_this_month = daily[
    daily['month'] == today_month]['daily_kwh'].mean()
avg_7_day_kwh      = daily['daily_kwh'].iloc[-7:].mean()
avg_14_day_kwh     = daily['daily_kwh'].iloc[-14:].mean()
avg_30_day_kwh     = daily['daily_kwh'].iloc[-30:].mean()

print(f"Predicting from today: {today.strftime('%A, %B %d')}")
print("Predicted daily bills for next 7 days (TOU tariff):")

bill_predictions = []
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

for i in range(1, 8):
    future_dow    = (today_dow + i) % 7
    future_month  = today_month
    pred_bill     = rf_model.predict([[
        future_dow, future_month,
        avg_kwh_this_month,
        avg_7_day_kwh,
        avg_14_day_kwh,
        avg_30_day_kwh
    ]])[0]

    # Calculate average tariff for this day type
    # (weighted by peak/off-peak hours ratio)
    if future_dow >= 5:
        avg_day_tariff = 0.12
        tariff_label   = 'Weekend rate (€0.12/kWh all day)'
    else:
        # 15 peak hours (07-22), 9 off-peak hours (22-07)
        avg_day_tariff = (15 * 0.20 + 9 * 0.10) / 24
        tariff_label   = 'Weekday (peak €0.20 / off-peak €0.10)'

    bill_predictions.append({
        'day':           days[future_dow],
        'predicted_bill': round(pred_bill, 2),
        'tariff_label':   tariff_label,
        'avg_tariff':     round(avg_day_tariff, 3),
    })

for i, p in enumerate(bill_predictions):
    print(f"  Day {i+1} ({p['day']}): "
          f"€{p['predicted_bill']:.2f}  "
          f"[{p['tariff_label']}]")

weekly_pred = sum(p['predicted_bill'] for p in bill_predictions)
print(f"\nPredicted weekly total:  €{weekly_pred:.2f}")
print(f"Predicted monthly total: €{weekly_pred * 4.33:.2f}")

# ── 4. Save results ───────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  SAVING RESULTS")
print("=" * 50)

anomalies[['datetime', 'total_power', 'expected_power',
           'power_deviation', 'anomaly_label']]\
    .to_csv('anomalies.csv', index=False)
print("Anomalies saved to: anomalies.csv")

daily[['date', 'daily_kwh', 'daily_bill']]\
    .to_csv('daily_summary.csv', index=False)
print("Daily summary saved to: daily_summary.csv")

# ── Save ML results to Firebase ───────────────────────────────────────────────
print("\nSaving ML results to Firebase...")
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase_key.json')
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    # Historical hourly averages by month and day-of-week
    hourly_profile = df.groupby(
        ['month', 'day_of_week', 'hour'])['total_power']\
        .mean().reset_index()
    hourly_profile.columns = ['month', 'dow', 'hour', 'avg_power']

    hourly_profile_list = [
        {
            'month':          int(row['month']),
            'dow':            int(row['dow']),
            'hour':           int(row['hour']),
            'avg_power':      round(float(row['avg_power']), 3),
            'tariff':         get_tariff(
                                  int(row['hour']),
                                  int(row['dow'])),
            'tariff_period':  get_tariff_period(
                                  int(row['hour']),
                                  int(row['dow'])),
        }
        for _, row in hourly_profile.iterrows()
    ]
    print(f"Hourly profile: {len(hourly_profile_list)} "
          f"month×dow×hour combinations calculated")

    # Tariff structure for Firebase
    tariff_structure = {
        'peak_rate':        0.20,
        'offpeak_rate':     0.10,
        'weekend_rate':     0.12,
        'peak_hours':       '07:00–22:00',
        'offpeak_hours':    '22:00–07:00',
        'peak_days':        'Monday–Friday',
        'weekend_days':     'Saturday–Sunday',
        'currency':         'EUR',
        'note':             'Time-of-use tariff — reflects common '
                            'European residential pricing structure',
    }

    ml_results = {
        'timestamp':        pd.Timestamp.now().isoformat(),
        'model_accuracy':   round(float(test_score), 3),
        'model_mae':        round(float(mae), 3),
        'model_rmse':       round(float(rmse), 3),
        'tariff_structure': tariff_structure,
        'predictions':      [
            {
                'day':            p['day'],
                'predicted_bill': p['predicted_bill'],
                'tariff_label':   p['tariff_label'],
                'avg_tariff':     p['avg_tariff'],
            }
            for p in bill_predictions
        ],
        'weekly_total':     round(weekly_pred, 2),
        'monthly_total':    round(weekly_pred * 4.33, 2),
        'hourly_profile':   hourly_profile_list,
    }

    db.collection('ml_results').document('latest').set(ml_results)
    print("ML results saved to Firebase!")
    print(f"\nTariff structure saved:")
    print(f"  Peak (weekday 07-22):     €0.20/kWh")
    print(f"  Off-peak (weekday 22-07): €0.10/kWh")
    print(f"  Weekend:                  €0.12/kWh")
    print(f"\nModel quality summary (TOU tariff):")
    print(f"  R²:   {test_score:.3f}")
    print(f"  MAE:  €{mae:.3f}/day")
    print(f"  RMSE: €{rmse:.3f}/day")

except Exception as e:
    print(f"Firebase save skipped: {e}")

print("\nDone!")