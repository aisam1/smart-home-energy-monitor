import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# ── Time-of-use tariff ────────────────────────────────────────────────────────
def get_tariff(hour, day_of_week):
    is_weekend = day_of_week >= 5
    if is_weekend:
        return 0.12
    elif 7 <= hour < 22:
        return 0.20
    else:
        return 0.10

def get_tariff_period(hour, day_of_week):
    if day_of_week >= 5:
        return 'Weekend'
    elif 7 <= hour < 22:
        return 'Peak'
    else:
        return 'Off-Peak'

def get_tariff_color(period):
    return {'Peak': '#e74c3c', 'Off-Peak': '#2ecc71',
            'Weekend': '#f39c12'}.get(period, '#3498db')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Home Energy Monitor",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: #1A1F2E;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2A3148;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #2A3148;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Connect to Firebase ───────────────────────────────────────────────────────
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase_key.json')
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firebase()

# ── Load UCI data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_uci_data():
    df = pd.read_csv('household_power_consumption.txt',
                     sep=';', low_memory=False, na_values=['?'])
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S')
    df['total_power'] = pd.to_numeric(
        df['Global_active_power'], errors='coerce')
    df['sub1'] = pd.to_numeric(df['Sub_metering_1'], errors='coerce')
    df['sub2'] = pd.to_numeric(df['Sub_metering_2'], errors='coerce')
    df['sub3'] = pd.to_numeric(df['Sub_metering_3'], errors='coerce')
    df = df.dropna(subset=['total_power'])
    df['sub1'] = df['sub1'].fillna(0)
    df['sub2'] = df['sub2'].fillna(0)
    df['sub3'] = df['sub3'].fillna(0)
    df['hour']        = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month']       = df['datetime'].dt.month
    df['date']        = df['datetime'].dt.date
    df['kwh']         = df['total_power'] * (1/60)
    df['kitchen_kwh'] = df['sub1'] / 1000
    df['laundry_kwh'] = df['sub2'] / 1000
    df['hvac_kwh']    = df['sub3'] / 1000
    df['other_kwh']   = (df['kwh'] - df['kitchen_kwh'] -
                         df['laundry_kwh'] - df['hvac_kwh']).clip(lower=0)
    df['tariff']        = df.apply(
        lambda r: get_tariff(r['hour'], r['day_of_week']), axis=1)
    df['tariff_period'] = df.apply(
        lambda r: get_tariff_period(r['hour'], r['day_of_week']), axis=1)
    df['cost']          = df['kwh'] * df['tariff']
    return df

# ── Data quality stats ────────────────────────────────────────────────────────
@st.cache_data
def get_data_quality(df_raw):
    df_raw2 = pd.read_csv('household_power_consumption.txt',
                          sep=';', low_memory=False, na_values=['?'])
    total_rows    = len(df_raw2)
    missing_power = df_raw2['Global_active_power'].isna().sum()
    missing_sub1  = df_raw2['Sub_metering_1'].isna().sum()
    missing_sub2  = df_raw2['Sub_metering_2'].isna().sum()
    missing_sub3  = df_raw2['Sub_metering_3'].isna().sum()
    df_raw2['datetime'] = pd.to_datetime(
        df_raw2['Date'] + ' ' + df_raw2['Time'],
        format='%d/%m/%Y %H:%M:%S', errors='coerce')
    date_min  = df_raw2['datetime'].min()
    date_max  = df_raw2['datetime'].max()
    span_days = (date_max - date_min).days
    expected  = span_days * 24 * 60
    coverage  = (total_rows / expected * 100) if expected > 0 else 0
    return {
        'total_rows':    total_rows,
        'missing_power': missing_power,
        'missing_sub1':  missing_sub1,
        'missing_sub2':  missing_sub2,
        'missing_sub3':  missing_sub3,
        'date_min':      date_min,
        'date_max':      date_max,
        'span_days':     span_days,
        'coverage_pct':  round(coverage, 1),
        'completeness':  round((1 - missing_power/total_rows)*100, 1),
    }

# ── Anomaly detection ─────────────────────────────────────────────────────────
@st.cache_data
def run_anomaly_detection(df):
    hourly_avg      = df.groupby('hour')['total_power'].mean()
    df              = df.copy()
    df['expected']  = df['hour'].map(hourly_avg)
    df['deviation'] = df['total_power'] - df['expected']
    features        = df[['total_power', 'deviation', 'hour']].values
    iso             = IsolationForest(contamination=0.03, random_state=42)
    df['anomaly']   = iso.fit_predict(features)
    return df

# ── Bill prediction ───────────────────────────────────────────────────────────
@st.cache_data
def run_bill_prediction_full(df_full_data):
    daily = df_full_data.groupby('date').agg(
        daily_kwh=('kwh',  'sum'),
        daily_bill=('cost', 'sum')
    ).reset_index()
    daily['date']        = pd.to_datetime(daily['date'])
    daily['day_of_week'] = daily['date'].dt.dayofweek
    daily['month']       = daily['date'].dt.month

    daily['prev_day_kwh']    = daily['daily_kwh'].shift(1)
    daily['prev_7_day_kwh']  = daily['daily_kwh'].rolling(7).mean()
    daily['prev_14_day_kwh'] = daily['daily_kwh'].rolling(14).mean()
    daily['prev_30_day_kwh'] = daily['daily_kwh'].rolling(30).mean()
    daily = daily.dropna()

    features = daily[['day_of_week', 'month', 'prev_day_kwh',
                       'prev_7_day_kwh', 'prev_14_day_kwh',
                       'prev_30_day_kwh']]
    target   = daily['daily_bill']

    split   = int(len(features) * 0.8)
    X_train = features.iloc[:split]
    X_test  = features.iloc[split:]
    y_train = target.iloc[:split]
    y_test  = target.iloc[split:]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred   = rf.predict(X_test)
    test_r2  = rf.score(X_test, y_test)
    train_r2 = rf.score(X_train, y_train)
    mae      = mean_absolute_error(y_test, y_pred)
    rmse     = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

    today       = datetime.now()
    today_dow   = today.weekday()
    today_month = today.month

    month_data      = daily[daily['month'] == today_month]
    avg_prev_day    = month_data['daily_kwh'].mean()
    avg_prev_7_day  = daily['daily_kwh'].iloc[-7:].mean()
    avg_prev_14_day = daily['daily_kwh'].iloc[-14:].mean()
    avg_prev_30_day = daily['daily_kwh'].iloc[-30:].mean()

    days  = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    preds = []
    for i in range(1, 8):
        future_dow   = (today_dow + i) % 7
        pred_bill    = rf.predict([[
            future_dow, today_month,
            avg_prev_day, avg_prev_7_day,
            avg_prev_14_day, avg_prev_30_day
        ]])[0]
        if future_dow >= 5:
            tariff_label = 'Weekend (€0.12/kWh all day)'
        else:
            tariff_label = 'Weekday (peak €0.20 / off-peak €0.10)'
        preds.append({
            'day':           days[future_dow],
            'predicted_bill': round(pred_bill, 2),
            'date':          pd.Timestamp(today) + pd.Timedelta(days=i),
            'tariff_label':  tariff_label,
        })

    return {
        'preds':    preds,
        'mae':      round(mae, 3),
        'rmse':     round(rmse, 3),
        'test_r2':  round(test_r2, 3),
        'train_r2': round(train_r2, 3),
    }

# ── LSTM predictions ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_lstm_predictions():
    try:
        if os.path.exists('lstm_predictions.csv'):
            pred_df    = pd.read_csv('lstm_predictions.csv')
            predictions = []
            for _, row in pred_df.iterrows():
                hour   = int(row.get('hour', 0))
                tariff = float(row['tariff']) if 'tariff' in pred_df.columns else 0.15
                period = str(row['tariff_period']) if 'tariff_period' in pred_df.columns else 'Unknown'
                cost   = float(row['predicted_cost']) if 'predicted_cost' in pred_df.columns \
                         else float(row['predicted_power']) * tariff
                predictions.append({
                    'hour':            hour,
                    'datetime':        str(row['datetime']),
                    'predicted_power': float(row['predicted_power']),
                    'predicted_cost':  cost,
                    'tariff':          tariff,
                    'tariff_period':   period,
                })
            total_kwh  = float(pred_df['predicted_power'].sum())
            total_cost = sum(p['predicted_cost'] for p in predictions)
            return {
                'mae':                  0.487,
                'rmse':                 0.637,
                'total_predicted_kwh':  round(total_kwh, 2),
                'total_predicted_cost': round(total_cost, 2),
                'predictions':          predictions,
            }
        docs = db.collection('lstm_predictions')\
                 .order_by('timestamp',
                           direction=firestore.Query.DESCENDING)\
                 .limit(1).stream()
        for doc in docs:
            return doc.to_dict()
        return None
    except Exception:
        return None

# ── Load data ─────────────────────────────────────────────────────────────────
df_full    = load_uci_data()
df_full    = run_anomaly_detection(df_full)
dq         = get_data_quality(df_full)
rf_results = run_bill_prediction_full(df_full)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

now          = datetime.now()
curr_tariff  = get_tariff(now.hour, now.weekday())
curr_period  = get_tariff_period(now.hour, now.weekday())

st.sidebar.markdown("### 💶 Electricity Tariff (TOU)")
st.sidebar.markdown("""
| Period | Hours | Days | Price |
|---|---|---|---|
| 🔴 Peak | 07:00–22:00 | Mon–Fri | €0.20/kWh |
| 🟢 Off-peak | 22:00–07:00 | Mon–Fri | €0.10/kWh |
| 🟡 Weekend | All day | Sat–Sun | €0.12/kWh |
""")
st.sidebar.markdown(
    f"**Now:** {curr_period} — €{curr_tariff:.2f}/kWh ({now.strftime('%H:%M')})")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Date Range Filter")
min_date = pd.to_datetime(df_full['date'].min())
max_date = pd.to_datetime(df_full['date'].max())

date_from = st.sidebar.date_input("From",
    value=max_date - pd.Timedelta(days=30),
    min_value=min_date, max_value=max_date)
date_to = st.sidebar.date_input("To",
    value=max_date,
    min_value=min_date, max_value=max_date)

st.sidebar.markdown("---")
st.sidebar.info(
    "📌 Date range filters historical charts only. "
    "ML predictions always use the full 4-year dataset.")

with st.sidebar.expander("ℹ️ About this Dashboard"):
    st.markdown("""
**Smart Home Energy Monitor**
Final year project - Electrical Engineering

**Dataset**
UCI ML Repository - Individual Household
Electric Power Consumption
- Source: Monitored French household
- Period: Dec 2006 - Nov 2010
- Frequency: 1 reading per minute
- Size: ~2 million readings

**ML Models Used**
- 🌲 **Isolation Forest** - Anomaly detection
- 🌳 **Random Forest** - 7-day bill prediction
- 🧠 **LSTM Neural Network** - 24h consumption forecast

**Tariff (TOU)**
- 🔴 Peak: €0.20/kWh (07-22 weekdays)
- 🟢 Off-peak: €0.10/kWh (22-07 weekdays)
- 🟡 Weekend: €0.12/kWh (all day)

**Tech Stack**
- Python, TensorFlow, scikit-learn
- Firebase Firestore
- Streamlit, Plotly
- Flutter (mobile app)

**Author:** Aiša Muhić
**Year:** 2026
""")

with st.sidebar.expander("📊 Dataset Quality"):
    st.metric("Total Readings", f"{dq['total_rows']:,}")
    st.metric("Date Coverage",
              f"{dq['date_min'].strftime('%Y-%m-%d')} → "
              f"{dq['date_max'].strftime('%Y-%m-%d')}")
    st.metric("Span", f"{dq['span_days']} days")
    st.metric("Data Completeness", f"{dq['completeness']}%")
    st.metric("Coverage", f"{dq['coverage_pct']}%")

# ── Filter data ───────────────────────────────────────────────────────────────
df = df_full[
    (df_full['date'] >= date_from) &
    (df_full['date'] <= date_to)
].copy()

days_selected = max((pd.to_datetime(date_to) -
                     pd.to_datetime(date_from)).days, 1)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("⚡ Smart Home Energy Monitor")
st.caption(
    f"Showing **{len(df):,}** readings from "
    f"**{date_from}** to **{date_to}** "
    f"({days_selected} days) · "
    f"TOU tariff: Peak €0.20 / Off-peak €0.10 / Weekend €0.12 · "
    f"ML predictions use full 4-year dataset")

# ── Top metrics ───────────────────────────────────────────────────────────────
total_kwh     = df['kwh'].sum()
total_cost    = df['cost'].sum()
avg_power     = df['total_power'].mean()
anomaly_count = len(df[df['anomaly'] == -1])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("⚡ Avg Power",     f"{avg_power:.2f} kW")
col2.metric("🔋 Total Energy",  f"{total_kwh:.1f} kWh")
col3.metric("💶 Total Cost",    f"€{total_cost:.2f}")
col4.metric("⚠️ Anomalies",     f"{anomaly_count}")
col5.metric("📅 Daily Average", f"€{total_cost/days_selected:.2f}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — POWER CONSUMPTION & ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("⚡ Power Consumption & Anomaly Detection")
st.caption("Hover over any point for details • Red × = anomaly • Drag to pan")

anomalies = df[df['anomaly'] == -1]
normal    = df[df['anomaly'] ==  1]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=normal['datetime'], y=normal['total_power'],
    mode='lines', name='Normal',
    line=dict(color='steelblue', width=1),
    hovertemplate=(
        '<b>%{x}</b><br>'
        'Power: %{y:.3f} kW<br>'
        'Tariff: €%{customdata[0]:.2f}/kWh (%{customdata[2]})<br>'
        'Cost/hr: €%{customdata[1]:.4f}<br>'
        'Expected: %{customdata[3]:.2f} kW<br>'
        'Status: ✅ Normal<extra></extra>'),
    customdata=np.column_stack([
        normal['tariff'],
        normal['total_power'] * normal['tariff'],
        normal['tariff_period'],
        normal['expected']])))

fig.add_trace(go.Scatter(
    x=anomalies['datetime'], y=anomalies['total_power'],
    mode='markers', name='⚠️ Anomaly',
    marker=dict(color='red', size=8, symbol='x'),
    hovertemplate=(
        '<b>%{x}</b><br>'
        'Power: %{y:.3f} kW<br>'
        'Expected: %{customdata[0]:.2f} kW<br>'
        'Deviation: +%{customdata[1]:.2f} kW<br>'
        'Tariff: €%{customdata[2]:.2f}/kWh (%{customdata[3]})<br>'
        'Status: ⚠️ Anomaly<extra></extra>'),
    customdata=np.column_stack([
        anomalies['expected'],
        anomalies['deviation'],
        anomalies['tariff'],
        anomalies['tariff_period']])))

fig.update_layout(
    height=380, xaxis_title='Time',
    yaxis_title='Power (kW)',
    legend=dict(x=0, y=1),
    dragmode='pan', hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)

with st.expander(f"🔍 View Anomaly Details ({anomaly_count} found)"):
    if len(anomalies) == 0:
        st.info("No anomalies in selected period.")
    else:
        anom_table = anomalies[[
            'datetime', 'total_power',
            'expected', 'deviation', 'tariff_period']].copy()
        anom_table['severity'] = pd.cut(
            anom_table['deviation'].abs(),
            bins=[0, 1, 3, float('inf')],
            labels=['⚠️ Warning', '🔶 High', '🔴 Critical'])
        anom_table['cost_impact'] = (
            anom_table['deviation'] * anomalies['tariff']).round(4)
        anom_table.columns = [
            'Timestamp', 'Actual (kW)', 'Expected (kW)',
            'Deviation (kW)', 'Tariff Period',
            'Severity', 'Cost Impact (€/hr)']
        anom_table = anom_table.sort_values(
            'Deviation (kW)', ascending=False)
        st.dataframe(anom_table.head(100), use_container_width=True)
        st.caption(
            f"Showing top 100 of {len(anomalies)} anomalies "
            f"sorted by deviation size")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DAILY & HOURLY PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Usage Patterns")

col_l, col_r = st.columns(2)

with col_l:
    st.markdown("**Daily Energy Cost (TOU tariff)**")
    daily_filtered = df.groupby('date').agg(
        bill=('cost', 'sum'),
        kwh=('kwh', 'sum'),
        avg_power=('total_power', 'mean')
    ).reset_index()
    daily_filtered['date'] = pd.to_datetime(daily_filtered['date'])

    fig_daily = go.Figure(go.Bar(
        x=daily_filtered['date'],
        y=daily_filtered['bill'],
        marker_color='steelblue',
        hovertemplate=(
            '<b>%{x|%Y-%m-%d}</b><br>'
            'Cost (TOU): €%{y:.2f}<br>'
            'Energy: %{customdata[0]:.2f} kWh<br>'
            'Avg Power: %{customdata[1]:.2f} kW'
            '<extra></extra>'),
        customdata=np.column_stack([
            daily_filtered['kwh'],
            daily_filtered['avg_power']])))
    fig_daily.update_layout(
        height=280, xaxis_title='Date',
        yaxis_title='Cost (€)', margin=dict(t=10))
    st.plotly_chart(fig_daily, use_container_width=True)

with col_r:
    st.markdown("**Average Usage by Hour + Tariff Period**")
    hourly_agg = df.groupby(['hour', 'tariff_period']).agg(
        avg_power=('total_power', 'mean'),
        tariff=('tariff', 'mean')
    ).reset_index()

    bar_colors = hourly_agg['tariff_period'].map(
        {'Peak': '#e74c3c',
         'Off-Peak': '#2ecc71',
         'Weekend': '#f39c12'}).fillna('#3498db')

    fig_hourly = go.Figure(go.Bar(
        x=hourly_agg['hour'],
        y=hourly_agg['avg_power'],
        marker_color=bar_colors,
        hovertemplate=(
            '<b>%{x}:00</b><br>'
            'Avg Power: %{y:.3f} kW<br>'
            'Tariff: €%{customdata[0]:.2f}/kWh<br>'
            'Period: %{customdata[1]}'
            '<extra></extra>'),
        customdata=np.column_stack([
            hourly_agg['tariff'],
            hourly_agg['tariff_period']])))
    fig_hourly.update_layout(
        height=280,
        xaxis_title='Hour of Day',
        yaxis_title='Avg Power (kW)',
        margin=dict(t=10))
    st.plotly_chart(fig_hourly, use_container_width=True)
    st.caption("🔴 Peak (€0.20)  |  🟢 Off-peak (€0.10)  |  🟡 Weekend (€0.12)")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PERIOD COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📈 Period Comparison")
st.caption("Compares selected period with the same duration immediately before it")

prev_date_to   = pd.to_datetime(date_from) - pd.Timedelta(days=1)
prev_date_from = prev_date_to - pd.Timedelta(days=days_selected)

df_prev = df_full[
    (df_full['date'] >= prev_date_from.date()) &
    (df_full['date'] <= prev_date_to.date())
].copy()

if len(df_prev) == 0:
    st.info("Not enough historical data for comparison.")
else:
    curr_avg  = df['total_power'].mean()
    prev_avg  = df_prev['total_power'].mean()
    curr_cost = df['cost'].sum()
    prev_cost = df_prev['cost'].sum()
    curr_kwh  = df['kwh'].sum()
    prev_kwh  = df_prev['kwh'].sum()

    diff_pct = ((curr_avg - prev_avg) / prev_avg * 100 if prev_avg > 0 else 0)
    improved = diff_pct < 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Power - Current",
              f"{curr_avg:.2f} kW",
              f"{diff_pct:+.1f}% vs previous",
              delta_color="inverse")
    c2.metric("Total Cost - Current",
              f"€{curr_cost:.2f}",
              f"€{curr_cost - prev_cost:+.2f} vs previous",
              delta_color="inverse")
    c3.metric("Total Energy - Current",
              f"{curr_kwh:.1f} kWh",
              f"{curr_kwh - prev_kwh:+.1f} kWh vs previous",
              delta_color="inverse")

    curr_hourly = df.groupby('hour')['total_power'].mean()
    prev_hourly = df_prev.groupby('hour')['total_power'].mean()

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=curr_hourly.index, y=curr_hourly.values,
        name=f'Selected ({date_from} to {date_to})',
        line=dict(color='steelblue', width=2),
        hovertemplate='Hour %{x}:00<br>Current: %{y:.3f} kW<extra></extra>'))
    fig_comp.add_trace(go.Scatter(
        x=prev_hourly.index, y=prev_hourly.values,
        name='Previous period',
        line=dict(color='orange', width=2, dash='dash'),
        hovertemplate='Hour %{x}:00<br>Previous: %{y:.3f} kW<extra></extra>'))
    fig_comp.update_layout(
        height=300,
        xaxis_title='Hour of Day',
        yaxis_title='Avg Power (kW)',
        legend=dict(x=0, y=1))
    st.plotly_chart(fig_comp, use_container_width=True)

    color  = "green" if improved else "orange"
    symbol = "📉" if improved else "📈"
    st.markdown(
        f":{color}[{symbol} **{'Improvement' if improved else 'Increase'}:** "
        f"Power usage {'decreased' if improved else 'increased'} by "
        f"**{abs(diff_pct):.1f}%** compared to the previous {days_selected}-day period.]")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ML PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🤖 ML Predictions")

tab1, tab2 = st.tabs([
    "📅 7-Day Bill Forecast (Random Forest)",
    "🧠 24h Hourly Forecast (LSTM)"
])

with tab1:
    st.caption(
        "Trained on **full 4-year UCI dataset** (2006–2010) · "
        "Predicts next 7 days from **today's date** · "
        "Costs reflect TOU tariff structure")

    preds   = rf_results['preds']
    mae     = rf_results['mae']
    rmse    = rf_results['rmse']
    test_r2 = rf_results['test_r2']
    pred_df = pd.DataFrame(preds)

    st.info(
        f"**Model Performance (evaluated on 2009–2010 test set):**  "
        f"MAE = **€{mae:.3f}/day** · "
        f"RMSE = **€{rmse:.3f}/day** · "
        f"R² = {test_r2:.3f}  \n"
        f"The model predicts daily electricity costs within ~€{mae:.2f} on average "
        f"using a time-of-use tariff (peak €0.20 / off-peak €0.10 / weekend €0.12). "
        f"R² is moderate by design — daily consumption is driven by unobservable "
        f"factors such as occupancy and weather. "
        f"The model's value lies in correctly ranking days for scheduling.")

    col_chart, col_stats = st.columns([3, 1])

    with col_chart:
        best_day  = pred_df.loc[pred_df['predicted_bill'].idxmin(), 'day']
        worst_day = pred_df.loc[pred_df['predicted_bill'].idxmax(), 'day']
        min_bill  = pred_df['predicted_bill'].min()
        max_bill  = pred_df['predicted_bill'].max()

        colors = []
        for _, row in pred_df.iterrows():
            if row['day'] == best_day:   colors.append('#2ecc71')
            elif row['day'] == worst_day: colors.append('#e74c3c')
            else:                         colors.append('#3498db')

        fig_pred = go.Figure(go.Bar(
            x=pred_df['day'],
            y=pred_df['predicted_bill'],
            marker_color=colors,
            hovertemplate=(
                '<b>%{x}</b><br>'
                'Predicted cost: €%{y:.2f}<br>'
                'Tariff: %{customdata[0]}<br>'
                'Date: %{customdata[1]}'
                '<extra></extra>'),
            customdata=np.column_stack([
                pred_df['tariff_label'],
                pred_df['date'].astype(str).str[:10]])))
        fig_pred.update_layout(
            height=280,
            xaxis_title='Day',
            yaxis_title='Predicted Cost (€)',
            margin=dict(t=10))
        st.plotly_chart(fig_pred, use_container_width=True)

        saving = max_bill - min_bill
        st.success(
            f"💡 **Scheduling tip:** Run water heater, washing machine "
            f"and oven on **{best_day}** for lowest weekly cost. "
            f"Avoid heavy usage on **{worst_day}**. "
            f"Potential saving: **€{saving:.2f}** by shifting "
            f"appliance use from {worst_day} to {best_day}. "
            f"Additional savings by using off-peak hours (22:00–07:00).")

    with col_stats:
        weekly  = pred_df['predicted_bill'].sum()
        monthly = weekly * 4.33
        st.metric("Weekly Total",  f"€{weekly:.2f}")
        st.metric("Monthly Est.",  f"€{monthly:.2f}")
        st.metric("RF MAE",        f"€{mae:.3f}/day")
        st.metric("RF RMSE",       f"€{rmse:.3f}/day")
        st.metric("Best Day 🟢",   best_day)
        st.metric("Worst Day 🔴",  worst_day)

    with st.expander("📋 Full 7-Day Prediction Table"):
        table = pred_df.copy()
        table['date']           = table['date'].astype(str).str[:10]
        table['predicted_bill'] = table['predicted_bill'].apply(
            lambda x: f"€{x:.2f}")
        table['vs_best']        = pred_df['predicted_bill'].apply(
            lambda x: f"+€{x - min_bill:.2f}" if x > min_bill else "✅ Cheapest")
        table = table[['day', 'predicted_bill', 'tariff_label', 'date', 'vs_best']]
        table.columns = ['Day', 'Predicted Bill', 'Tariff', 'Date', 'vs Best Day']
        st.dataframe(table, use_container_width=True)

with tab2:
    lstm_data = load_lstm_predictions()

    if lstm_data is None:
        st.info("No LSTM predictions found. Run `lstm_model.py` to generate predictions.")
    else:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("MAE",            f"{lstm_data.get('mae',0):.3f} kW")
        col_m2.metric("RMSE",           f"{lstm_data.get('rmse',0):.3f} kW")
        col_m3.metric("24h Energy",     f"{lstm_data.get('total_predicted_kwh',0):.1f} kWh")
        col_m4.metric("24h Cost (TOU)", f"€{lstm_data.get('total_predicted_cost',0):.2f}")

        predictions = lstm_data.get('predictions', [])
        if predictions:
            pred_lstm = pd.DataFrame(predictions)
            pred_lstm['datetime'] = pd.to_datetime(pred_lstm['datetime'])

            if 'tariff' not in pred_lstm.columns:
                pred_lstm['tariff'] = pred_lstm.apply(
                    lambda r: get_tariff(r['datetime'].hour, r['datetime'].dayofweek), axis=1)
                pred_lstm['tariff_period'] = pred_lstm.apply(
                    lambda r: get_tariff_period(r['datetime'].hour, r['datetime'].dayofweek), axis=1)

            pred_lstm['predicted_cost'] = pred_lstm['predicted_power'] * pred_lstm['tariff']

            col_lc, col_lt = st.columns([3, 2])

            with col_lc:
                st.markdown("**Hourly Power Forecast**")

                fig_lstm = go.Figure()

                # Background shading — use rgba() not 8-digit hex
                for i, row in pred_lstm.iterrows():
                    period = row.get('tariff_period', 'Unknown')
                    color  = ('rgba(231,76,60,0.08)'   if period == 'Peak'
                              else 'rgba(46,204,113,0.08)' if period == 'Off-Peak'
                              else 'rgba(243,156,18,0.08)')
                    fig_lstm.add_vrect(
                        x0=row['datetime'] - pd.Timedelta(minutes=30),
                        x1=row['datetime'] + pd.Timedelta(minutes=30),
                        fillcolor=color, opacity=1,
                        layer='below', line_width=0)

                fig_lstm.add_trace(go.Scatter(
                    x=pred_lstm['datetime'],
                    y=pred_lstm['predicted_power'],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='mediumpurple', width=2),
                    marker=dict(size=5),
                    hovertemplate=(
                        '<b>%{x|%H:%M}</b><br>'
                        'Power: %{y:.3f} kW<br>'
                        'Cost: €%{customdata[0]:.4f}<br>'
                        'Tariff: €%{customdata[1]:.2f}/kWh (%{customdata[2]})'
                        '<extra></extra>'),
                    customdata=np.column_stack([
                        pred_lstm['predicted_cost'],
                        pred_lstm['tariff'],
                        pred_lstm['tariff_period']])))

                fig_lstm.update_layout(
                    height=280,
                    xaxis_title='Hour',
                    yaxis_title='Power (kW)',
                    margin=dict(t=10))
                st.plotly_chart(fig_lstm, use_container_width=True)
                st.caption("🔴 Peak (€0.20)  |  🟢 Off-peak (€0.10)  |  🟡 Weekend (€0.12)")

                peak_h = pred_lstm.loc[pred_lstm['predicted_power'].idxmax()]
                low_h  = pred_lstm.loc[pred_lstm['predicted_power'].idxmin()]
                st.info(
                    f"⚡ **Peak:** {peak_h['datetime'].strftime('%H:%M')} "
                    f"— {peak_h['predicted_power']:.2f} kW "
                    f"({peak_h.get('tariff_period','')}, €{peak_h.get('tariff',0.15):.2f}/kWh)"
                    f"   |   "
                    f"🌙 **Lowest:** {low_h['datetime'].strftime('%H:%M')} "
                    f"— {low_h['predicted_power']:.2f} kW "
                    f"({low_h.get('tariff_period','')}, €{low_h.get('tariff',0.15):.2f}/kWh)")

            with col_lt:
                st.markdown("**Hourly Cost Forecast (TOU)**")
                table_df = pred_lstm[[
                    'datetime', 'predicted_power',
                    'predicted_cost', 'tariff_period']].copy()
                table_df['datetime'] = table_df['datetime'].dt.strftime('%H:%M')
                table_df['advice']   = table_df['tariff_period'].apply(
                    lambda p: '🔴 Peak rate'  if p == 'Peak'
                    else '🟢 Off-peak'        if p == 'Off-Peak'
                    else '🟡 Weekend')
                table_df.columns = ['Hour', 'Power (kW)', 'Cost (€)', 'Period', 'Advice']
                table_df['Power (kW)'] = table_df['Power (kW)'].round(3)
                table_df['Cost (€)']   = table_df['Cost (€)'].round(4)
                st.dataframe(table_df, use_container_width=True, height=280)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — APPLIANCE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🔌 Real Appliance Cost Breakdown")
st.caption("Based on UCI sub-metering sensors · Costs use TOU tariff · Updates with date range")

kitchen_cost = df['kitchen_kwh'].mul(df['tariff']).sum()
laundry_cost = df['laundry_kwh'].mul(df['tariff']).sum()
hvac_cost    = df['hvac_kwh'].mul(df['tariff']).sum()
other_cost   = df['other_kwh'].mul(df['tariff']).sum()

app_data = pd.DataFrame({
    'Appliance': [
        'Kitchen (Oven/Dishwasher)',
        'Laundry (Washing Machine/Fridge)',
        'Water Heater & AC',
        'Other (Lighting/TV/Standby)'
    ],
    'Cost (€)': [kitchen_cost, laundry_cost, hvac_cost, other_cost],
    'kWh': [
        df['kitchen_kwh'].sum(),
        df['laundry_kwh'].sum(),
        df['hvac_kwh'].sum(),
        df['other_kwh'].sum()]
})

col_pie, col_detail = st.columns(2)

with col_pie:
    fig_pie = go.Figure(go.Pie(
        labels=app_data['Appliance'],
        values=app_data['Cost (€)'],
        hovertemplate=(
            '<b>%{label}</b><br>'
            'Cost (TOU): €%{value:.2f}<br>'
            'Share: %{percent}<br>'
            'Energy: %{customdata:.2f} kWh'
            '<extra></extra>'),
        customdata=app_data['kWh']))
    fig_pie.update_layout(height=300, margin=dict(t=10))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_detail:
    total_app = app_data['Cost (€)'].sum()
    for _, row in app_data.iterrows():
        pct = row['Cost (€)'] / total_app * 100 if total_app > 0 else 0
        st.markdown(
            f"**{row['Appliance']}**  "
            f"€{row['Cost (€)']:.2f} · "
            f"{row['kWh']:.1f} kWh · "
            f"{pct:.1f}%")
        st.progress(pct / 100)
    st.markdown(f"**Total: €{total_app:.2f}**")
    st.caption(
        "Sub_metering_1=Kitchen | "
        "Sub_metering_2=Laundry | "
        "Sub_metering_3=Water Heater/AC | "
        "Costs use TOU tariff per reading")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — BILL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("💶 Bill Summary (TOU Tariff)")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
col_s1.metric("Selected Period",  f"€{total_cost:.2f}")
col_s2.metric("Daily Average",    f"€{total_cost/days_selected:.2f}")
col_s3.metric("Monthly Estimate", f"€{total_cost/days_selected*30:.2f}")
col_s4.metric("Yearly Estimate",  f"€{total_cost/days_selected*365:.2f}")

st.markdown("**Cost breakdown by tariff period:**")
t1, t2, t3 = st.columns(3)
peak_cost    = df[df['tariff_period'] == 'Peak']['cost'].sum()
offpeak_cost = df[df['tariff_period'] == 'Off-Peak']['cost'].sum()
weekend_cost = df[df['tariff_period'] == 'Weekend']['cost'].sum()
t1.metric("🔴 Peak hours",     f"€{peak_cost:.2f}")
t2.metric("🟢 Off-peak hours", f"€{offpeak_cost:.2f}")
t3.metric("🟡 Weekend",        f"€{weekend_cost:.2f}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🔬 Dataset Quality Report"):
    st.markdown("### UCI Household Power Consumption Dataset")
    q1, q2, q3 = st.columns(3)
    q1.metric("Total Readings", f"{dq['total_rows']:,}")
    q2.metric("Date Span",      f"{dq['span_days']} days")
    q3.metric("Completeness",   f"{dq['completeness']}%")

    q4, q5, q6 = st.columns(3)
    q4.metric("Coverage",      f"{dq['coverage_pct']}%")
    q5.metric("Missing Power", f"{dq['missing_power']:,}")
    q6.metric("Start → End",
              f"{dq['date_min'].strftime('%Y-%m')} → "
              f"{dq['date_max'].strftime('%Y-%m')}")

    st.markdown("**Missing values per sensor:**")
    missing_df = pd.DataFrame({
        'Sensor': [
            'Global Active Power',
            'Sub_metering_1 (Kitchen)',
            'Sub_metering_2 (Laundry)',
            'Sub_metering_3 (HVAC)'],
        'Missing': [
            dq['missing_power'], dq['missing_sub1'],
            dq['missing_sub2'], dq['missing_sub3']],
        'Missing %': [
            round(dq['missing_power']/dq['total_rows']*100, 2),
            round(dq['missing_sub1']/dq['total_rows']*100,  2),
            round(dq['missing_sub2']/dq['total_rows']*100,  2),
            round(dq['missing_sub3']/dq['total_rows']*100,  2)]
    })
    st.dataframe(missing_df, use_container_width=True)
    st.caption(
        "Missing values in sub-metering filled with 0. "
        "Rows with missing Global Active Power dropped (1.25%).")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📥 Export Data")

col_e1, col_e2, col_e3, col_e4 = st.columns(4)

with col_e1:
    hourly_exp = df.groupby(['hour', 'tariff_period']).agg(
        avg_power=('total_power', 'mean'),
        total_kwh=('kwh', 'sum'),
        total_cost=('cost', 'sum'),
        tariff=('tariff', 'mean')
    ).reset_index()
    st.download_button(
        label="⬇️ Hourly Summary",
        data=hourly_exp.to_csv(index=False).encode(),
        file_name=f"hourly_{date_from}_{date_to}.csv",
        mime='text/csv')

with col_e2:
    # Build daily export from current filtered df — avoids reference errors
    daily_export = df.groupby('date').agg(
        bill=('cost', 'sum'),
        kwh=('kwh', 'sum'),
        avg_power=('total_power', 'mean')
    ).reset_index() if len(df) > 0 else pd.DataFrame()
    st.download_button(
        label="⬇️ Daily Summary",
        data=daily_export.to_csv(index=False).encode(),
        file_name=f"daily_{date_from}_{date_to}.csv",
        mime='text/csv')

with col_e3:
    anom_exp = df[df['anomaly'] == -1][[
        'datetime', 'total_power', 'expected',
        'deviation', 'tariff_period']].copy()
    st.download_button(
        label="⬇️ Anomalies",
        data=anom_exp.to_csv(index=False).encode(),
        file_name=f"anomalies_{date_from}_{date_to}.csv",
        mime='text/csv')

with col_e4:
    report_lines = [
        f"Smart Home Energy Report",
        f"Period: {date_from} to {date_to}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Tariff: Peak €0.20 / Off-peak €0.10 / Weekend €0.12",
        f"",
        f"SUMMARY",
        f"Average Power,{avg_power:.2f} kW",
        f"Total Energy,{total_kwh:.2f} kWh",
        f"Total Cost (TOU),€{total_cost:.2f}",
        f"Peak Cost,€{peak_cost:.2f}",
        f"Off-peak Cost,€{offpeak_cost:.2f}",
        f"Weekend Cost,€{weekend_cost:.2f}",
        f"Daily Average,€{total_cost/days_selected:.2f}",
        f"Monthly Estimate,€{total_cost/days_selected*30:.2f}",
        f"Yearly Estimate,€{total_cost/days_selected*365:.2f}",
        f"Anomalies Detected,{anomaly_count}",
        f"",
        f"ML MODEL PERFORMANCE",
        f"Random Forest MAE,€{rf_results['mae']:.3f}/day",
        f"Random Forest RMSE,€{rf_results['rmse']:.3f}/day",
        f"Random Forest R²,{rf_results['test_r2']:.3f}",
        f"",
        f"APPLIANCE BREAKDOWN (TOU)",
        f"Kitchen,€{kitchen_cost:.2f}",
        f"Laundry,€{laundry_cost:.2f}",
        f"Water Heater/AC,€{hvac_cost:.2f}",
        f"Other,€{other_cost:.2f}",
    ]
    st.download_button(
        label="⬇️ Full Report",
        data='\n'.join(report_lines).encode(),
        file_name=f"energy_report_{date_from}_{date_to}.csv",
        mime='text/csv')

with st.expander("📋 Raw Data (last 50 readings)"):
    st.dataframe(
        df[['datetime', 'total_power', 'kwh', 'cost',
            'tariff', 'tariff_period', 'anomaly']].tail(50),
        use_container_width=True)