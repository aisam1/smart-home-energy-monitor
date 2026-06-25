import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import firebase_admin
from firebase_admin import credentials, firestore
import random
import warnings
warnings.filterwarnings('ignore')

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

print(f"TensorFlow version: {tf.__version__}")
print("Loading UCI dataset...")

# ── 1. Load and prepare data ──────────────────────────────────────────────────
df = pd.read_csv('household_power_consumption.txt',
                 sep=';', low_memory=False, na_values=['?'])

df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                format='%d/%m/%Y %H:%M:%S')
df['power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
df = df.dropna(subset=['power'])
df = df.set_index('datetime')

# Resample to hourly averages
hourly = df['power'].resample('h').mean().dropna()
print(f"Hourly data points: {len(hourly)}")

# ── 2. Feature engineering ────────────────────────────────────────────────────
data = pd.DataFrame({
    'power':       hourly.values,
    'hour':        hourly.index.hour,
    'day_of_week': hourly.index.dayofweek,
    'month':       hourly.index.month,
})

# ── 3. Scale data ─────────────────────────────────────────────────────────────
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data[['power']])

power_scaler = MinMaxScaler(feature_range=(0, 1))
power_scaler.fit(data[['power']])

# ── 4. Create sequences for LSTM ──────────────────────────────────────────────
SEQ_LENGTH  = 24
PRED_LENGTH = 24

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_len])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, SEQ_LENGTH, PRED_LENGTH)
print(f"Sequences created: {len(X)}")

split   = int(len(X) * 0.8)
X_train = X[:split]
X_test  = X[split:]
y_train = y[:split]
y_test  = y[split:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── 5. Build LSTM model ───────────────────────────────────────────────────────
print("\nBuilding LSTM model...")

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(PRED_LENGTH)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ── 6. Train model ────────────────────────────────────────────────────────────
print("\nTraining LSTM model...")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ── 7. Evaluate model ─────────────────────────────────────────────────────────
print("\nEvaluating model...")

y_pred_scaled = model.predict(X_test)
y_pred        = power_scaler.inverse_transform(y_pred_scaled)
y_true        = power_scaler.inverse_transform(
    y_test.reshape(-1, PRED_LENGTH))

mae  = mean_absolute_error(y_true.flatten(), y_pred.flatten())
rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))

print(f"\nModel Performance:")
print(f"  MAE  (Mean Absolute Error): {mae:.3f} kW")
print(f"  RMSE (Root Mean Sq Error):  {rmse:.3f} kW")

# ── 7b. Evaluation plots ──────────────────────────────────────────────────────
print("\nGenerating LSTM evaluation plots...")

y_pred_flat = y_pred.flatten()
y_true_flat = y_true.flatten()

# Plot 1 — Actual vs Predicted + scatter
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(y_true_flat[:168],
             label='Actual', color='steelblue', linewidth=1.5)
axes[0].plot(y_pred_flat[:168],
             label='Predicted', color='orange',
             linewidth=1.5, alpha=0.8)
axes[0].set_title(
    f'LSTM: Actual vs Predicted — First Week of Test Set\n'
    f'MAE = {mae:.3f} kW  |  RMSE = {rmse:.3f} kW')
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Power (kW)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_true_flat[:500], y_pred_flat[:500],
                alpha=0.3, color='mediumpurple', s=10)
axes[1].plot(
    [y_true_flat.min(), y_true_flat.max()],
    [y_true_flat.min(), y_true_flat.max()],
    'r--', linewidth=2, label='Perfect prediction (y=x)')
axes[1].set_title(
    'LSTM: Predicted vs Actual Scatter (first 500 test points)')
axes[1].set_xlabel('Actual Power (kW)')
axes[1].set_ylabel('Predicted Power (kW)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: lstm_evaluation.png")

# Plot 2 — Error distribution
errors = y_true_flat - y_pred_flat

fig2, ax = plt.subplots(figsize=(10, 4))
ax.hist(errors, bins=60, color='mediumpurple',
        alpha=0.7, edgecolor='white')
ax.axvline(0, color='red', linestyle='--',
           linewidth=2, label='Zero error')
ax.axvline(errors.mean(), color='orange', linestyle='--',
           linewidth=2,
           label=f'Mean error: {errors.mean():.3f} kW')
ax.set_title(
    f'LSTM Prediction Error Distribution\n'
    f'Std = {errors.std():.3f} kW  |  '
    f'Within ±0.5 kW: '
    f'{(np.abs(errors) < 0.5).mean()*100:.1f}% of predictions')
ax.set_xlabel('Error (Actual − Predicted) in kW')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lstm_error_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: lstm_error_distribution.png")

print(f"\nDetailed evaluation stats:")
print(f"  MAE:                 {mae:.3f} kW")
print(f"  RMSE:                {rmse:.3f} kW")
print(f"  Mean error (bias):   {errors.mean():.4f} kW")
print(f"  Error std:           {errors.std():.3f} kW")
print(f"  Within ±0.5 kW:      "
      f"{(np.abs(errors) < 0.5).mean()*100:.1f}% of predictions")
print(f"  Within ±1.0 kW:      "
      f"{(np.abs(errors) < 1.0).mean()*100:.1f}% of predictions")

# ── 8. Predict next 24 hours ──────────────────────────────────────────────────
print("\nPredicting next 24 hours...")

from datetime import datetime

today        = datetime.now()
target_month = today.month
target_dow   = today.weekday()

print(f"\nToday: {today.strftime('%A, %B %d')} "
      f"(month={target_month}, dow={target_dow})")
print("Finding matching historical windows...")

# Context-aware window selection
matching_starts = []
for i in range(len(scaled) - SEQ_LENGTH - PRED_LENGTH - 1):
    window_time = hourly.index[i]
    if (window_time.month == target_month and
            window_time.weekday() == target_dow):
        matching_starts.append(i)

print(f"Found {len(matching_starts)} matching windows "
      f"for {today.strftime('%B')} {today.strftime('%A')}s")

if len(matching_starts) < 5:
    print("Not enough day+month matches, "
          "falling back to month only...")
    matching_starts = []
    for i in range(len(scaled) - SEQ_LENGTH - PRED_LENGTH - 1):
        window_time = hourly.index[i]
        if window_time.month == target_month:
            matching_starts.append(i)
    print(f"Found {len(matching_starts)} matching windows "
          f"for {today.strftime('%B')} only")

if len(matching_starts) < 5:
    print("Falling back to day of week only...")
    matching_starts = []
    for i in range(len(scaled) - SEQ_LENGTH - PRED_LENGTH - 1):
        window_time = hourly.index[i]
        if window_time.weekday() == target_dow:
            matching_starts.append(i)
    print(f"Found {len(matching_starts)} matching windows "
          f"for {today.strftime('%A')}s")

random_start = random.choice(matching_starts) if matching_starts \
    else random.randint(
        0, len(scaled) - SEQ_LENGTH - PRED_LENGTH - 1)

# Average predictions from up to 5 similar windows for stability
sample_starts = random.sample(
    matching_starts,
    min(5, len(matching_starts))) if matching_starts \
    else [random_start]

all_predictions = []
for start in sample_starts:
    seq  = scaled[start:start + SEQ_LENGTH].reshape(1, SEQ_LENGTH, 1)
    pred = model.predict(seq, verbose=0)
    all_predictions.append(
        power_scaler.inverse_transform(pred)[0])

next_24h = np.mean(all_predictions, axis=0)

random_time = hourly.index[random_start]
last_time   = hourly.index[random_start + SEQ_LENGTH - 1]
pred_times  = pd.date_range(
    start=today.replace(hour=0, minute=0, second=0, microsecond=0),
    periods=PRED_LENGTH,
    freq='h'
)

actual_start = random_start + SEQ_LENGTH
actual_next  = power_scaler.inverse_transform(
    scaled[actual_start:actual_start + PRED_LENGTH]
    .reshape(-1, 1)).flatten()

print(f"\nContext: {today.strftime('%A')} in "
      f"{today.strftime('%B')} "
      f"— using {len(sample_starts)} historical windows")
print(f"Predicting for: {pred_times[0].strftime('%Y-%m-%d %H:%M')} "
      f"to {pred_times[-1].strftime('%Y-%m-%d %H:%M')}")

# ── Build prediction dataframe with TOU tariff ────────────────────────────────
predictions_list = []
total_predicted_kwh  = 0.0
total_predicted_cost = 0.0

print(f"\nNext 24 hours prediction (TOU tariff):")
print(f"{'Hour':<8} {'Power (kW)':<14} {'Tariff':<10} "
      f"{'Period':<12} {'Cost (€)'}")
print("-" * 55)

for i in range(PRED_LENGTH):
    hour         = pred_times[i].hour
    dow          = pred_times[i].dayofweek
    tariff       = get_tariff(hour, dow)
    period       = get_tariff_period(hour, dow)
    pred_power   = float(next_24h[i])
    pred_kwh     = pred_power * 1          # 1 hour
    pred_cost    = pred_kwh * tariff

    total_predicted_kwh  += pred_kwh
    total_predicted_cost += pred_cost

    predictions_list.append({
        'hour':            i,
        'datetime':        str(pred_times[i]),
        'predicted_power': round(pred_power, 3),
        'predicted_kwh':   round(pred_kwh, 3),
        'predicted_cost':  round(pred_cost, 4),
        'tariff':          tariff,
        'tariff_period':   period,
    })

    print(f"{hour:02d}:00    "
          f"{pred_power:.3f} kW      "
          f"€{tariff:.2f}      "
          f"{period:<12} "
          f"€{pred_cost:.4f}")

print("-" * 55)
print(f"Total predicted:  "
      f"{total_predicted_kwh:.2f} kWh  |  "
      f"€{total_predicted_cost:.2f}")

pred_df = pd.DataFrame(predictions_list)

# ── 9. Plot results ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Chart 1 — Training history
axes[0].plot(history.history['loss'],
             label='Training Loss', color='steelblue')
axes[0].plot(history.history['val_loss'],
             label='Validation Loss', color='orange')
axes[0].set_title('LSTM Training History')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()

# Chart 2 — Actual vs Predicted for today's pattern
time_range = range(PRED_LENGTH)
axes[1].plot(time_range, actual_next,
             label='Actual (historical)',
             color='steelblue', linewidth=2)
axes[1].plot(time_range, next_24h,
             label='Predicted', color='orange',
             linewidth=2, linestyle='--')

# Shade peak/off-peak/weekend hours
for i in range(PRED_LENGTH):
    hour   = pred_times[i].hour
    dow    = pred_times[i].dayofweek
    period = get_tariff_period(hour, dow)
    color  = ('#ff000015' if period == 'Peak'
              else '#00ff0015' if period == 'Off-Peak'
              else '#ffa50015')
    axes[1].axvspan(i - 0.5, i + 0.5, alpha=0.3,
                    color=color, linewidth=0)

axes[1].set_title(
    f'LSTM: Actual vs Predicted — '
    f'{today.strftime("%A, %B")} pattern '
    f'({len(sample_starts)} historical windows averaged)\n'
    f'🔴 Peak €0.20  🟢 Off-peak €0.10  🟡 Weekend €0.12')
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Power (kW)')
axes[1].legend()

plt.tight_layout()
plt.savefig('lstm_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nChart saved as lstm_predictions.png")

# ── 10. Save predictions ──────────────────────────────────────────────────────
print("\nSaving predictions...")

pred_df.to_csv('lstm_predictions.csv', index=False)
print("Predictions saved to lstm_predictions.csv")

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase_key.json')
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    lstm_results = {
        'timestamp':            pd.Timestamp.now().isoformat(),
        'mae':                  round(float(mae), 3),
        'rmse':                 round(float(rmse), 3),
        'context_month':        target_month,
        'context_dow':          target_dow,
        'context_day_name':     today.strftime('%A'),
        'context_month_name':   today.strftime('%B'),
        'windows_used':         len(sample_starts),
        'context_description':  (
            f"Based on {len(sample_starts)} historical "
            f"{today.strftime('%A')}s in "
            f"{today.strftime('%B')} "
            f"({today.strftime('%Y-%m-%d')})"),
        'total_predicted_kwh':  round(float(total_predicted_kwh), 2),
        'total_predicted_cost': round(float(total_predicted_cost), 2),
        'tariff_structure': {
            'peak_rate':     0.20,
            'offpeak_rate':  0.10,
            'weekend_rate':  0.12,
            'peak_hours':    '07:00–22:00',
            'offpeak_hours': '22:00–07:00',
            'peak_days':     'Monday–Friday',
            'weekend_days':  'Saturday–Sunday',
        },
        'predictions': [
            {
                'hour':            p['hour'],
                'datetime':        p['datetime'],
                'predicted_power': p['predicted_power'],
                'predicted_cost':  p['predicted_cost'],
                'tariff':          p['tariff'],
                'tariff_period':   p['tariff_period'],
            }
            for p in predictions_list
        ]
    }

    db.collection('lstm_predictions').add(lstm_results)
    print("LSTM predictions saved to Firebase!")
    print(f"\nTariff breakdown for today's forecast:")
    peak_hours    = [p for p in predictions_list
                     if p['tariff_period'] == 'Peak']
    offpeak_hours = [p for p in predictions_list
                     if p['tariff_period'] == 'Off-Peak']
    weekend_hours = [p for p in predictions_list
                     if p['tariff_period'] == 'Weekend']
    if peak_hours:
        print(f"  Peak hours (€0.20):     "
              f"{len(peak_hours)}h  →  "
              f"€{sum(p['predicted_cost'] for p in peak_hours):.2f}")
    if offpeak_hours:
        print(f"  Off-peak hours (€0.10): "
              f"{len(offpeak_hours)}h  →  "
              f"€{sum(p['predicted_cost'] for p in offpeak_hours):.2f}")
    if weekend_hours:
        print(f"  Weekend hours (€0.12):  "
              f"{len(weekend_hours)}h  →  "
              f"€{sum(p['predicted_cost'] for p in weekend_hours):.2f}")

except Exception as e:
    print(f"Firebase save skipped (CSV already saved): {e}")

print("\nDone! LSTM model complete.")