# app.py - Complete Vizag Traffic Predictor (Cloud-Ready, No matplotlib)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🚦 Vizag Traffic Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- FORCE LIGHT THEME (Fixes Black Background on Cloud) ---
st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    .st-bb, .st-at { background-color: #f0f2f6; }
    div[data-testid="stDataFrame"] { background-color: #ffffff; }
    .st-cb { background-color: #ffffff; }
    .st-dc { background-color: #ffffff; }
    .st-bx { background-color: #ffffff; }
    .st-ae { background-color: #ffffff; }
    .st-cd { background-color: #ffffff; }
    .st-df { background-color: #ffffff; }
    [data-testid="stAppViewContainer"] { background-color: #ffffff; }
    [data-testid="stHeader"] { background-color: #ffffff; }
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .traffic-card {
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .high-risk { background-color: #ffcccc; border-left: 8px solid #dc3545; }
    .medium-risk { background-color: #ffe5b4; border-left: 8px solid #fd7e14; }
    .low-risk { background-color: #d4edda; border-left: 8px solid #28a745; }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA GENERATION FUNCTION (Built-in) ---
def generate_vizag_data():
    locations = {
        "Gajuwaka": 450, "NAD Junction": 420, "Maddilapalem": 380,
        "Siripuram": 400, "MVP Colony": 350, "Dwaraka Nagar": 390,
        "RTC Complex": 500, "Jagadamba Junction": 480,
        "Akkayyapalem": 370, "Madhurawada": 280
    }
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 8, 10)
    date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    all_rows = []

    for single_date in date_list:
        for loc_name, base_traffic in locations.items():
            for hour in range(6, 23):
                day_name = single_date.strftime('%A')
                is_weekend = 1 if day_name in ['Saturday', 'Sunday'] else 0
                is_holiday = 0
                if single_date.month == 1 and single_date.day == 26: is_holiday = 1
                if single_date.month == 8 and single_date.day == 15: is_holiday = 1
                if single_date.month == 10 and single_date.day == 2: is_holiday = 1
                if single_date.month == 12 and single_date.day == 25: is_holiday = 1
                if random.random() < 0.02: is_holiday = 1

                if 8 <= hour <= 10: time_mult = 2.2
                elif 17 <= hour <= 19: time_mult = 2.5
                elif 12 <= hour <= 14: time_mult = 1.3
                elif 21 <= hour <= 22: time_mult = 0.6
                else: time_mult = 0.9

                if is_weekend or is_holiday:
                    if 10 <= hour <= 20: time_mult = 1.4
                    else: time_mult = 0.7

                temperature = round(np.random.normal(30, 4), 1)
                if single_date.month in [6, 7, 8, 9]:
                    rainfall = round(max(0, np.random.exponential(8)), 1)
                else:
                    rainfall = round(max(0, np.random.exponential(2)), 1)
                if rainfall > 15: time_mult *= 0.8

                vehicle_count = int(max(20, base_traffic * time_mult + np.random.normal(0, 40)))
                if vehicle_count > 700: avg_speed = round(max(5, np.random.normal(22, 5)), 1)
                elif vehicle_count > 400: avg_speed = round(max(10, np.random.normal(38, 7)), 1)
                else: avg_speed = round(max(15, np.random.normal(52, 10)), 1)
                avg_speed = round(max(5, avg_speed - rainfall * 0.3), 1)

                previous_traffic = int(max(50, vehicle_count + np.random.normal(0, 30)))
                previous_hour_traffic = int(max(50, vehicle_count + np.random.normal(0, 20)))

                all_rows.append({
                    'City': 'Visakhapatnam', 'Location': loc_name, 'Date': single_date.strftime('%Y-%m-%d'),
                    'Time': f"{hour:02d}:00", 'Hour': hour, 'Day': day_name,
                    'Is_Weekend': is_weekend, 'Is_Holiday': is_holiday,
                    'Temperature': temperature, 'Rainfall_mm': rainfall,
                    'Previous_Traffic': previous_traffic, 'Previous_Hour_Traffic': previous_hour_traffic,
                    'Vehicle_Count': vehicle_count, 'Avg_Speed_kmh': avg_speed
                })
    df = pd.DataFrame(all_rows)
    df.to_csv('vizag_traffic.csv', index=False)
    return df

# --- 2. MODEL TRAINING FUNCTION (Built-in) ---
def train_vizag_model():
    df = pd.read_csv('vizag_traffic.csv')
    features = ['Location', 'Hour', 'Day', 'Is_Weekend', 'Is_Holiday', 
                'Temperature', 'Rainfall_mm', 'Previous_Traffic', 'Previous_Hour_Traffic']
    target = 'Vehicle_Count'
    
    label_encoders = {}
    for col in ['Location', 'Day']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    pickle.dump(model, open('traffic_model.pkl', 'wb'))
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    pickle.dump(label_encoders, open('label_encoders.pkl', 'wb'))
    pickle.dump(features, open('feature_names.pkl', 'wb'))
    return model

# --- 3. AUTO-BUILD LOGIC (Runs on first launch) ---
if not os.path.exists('traffic_model.pkl'):
    st.warning("⚙️ First time setup! Generating Vizag data and training model... (~2 minutes)")
    generate_vizag_data()
    train_vizag_model()
    st.success("✅ Model ready! Refreshing...")
    st.rerun()

# --- 4. LOAD MODELS ---
@st.cache_resource
def load_models():
    model = pickle.load(open('traffic_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    encoders = pickle.load(open('label_encoders.pkl', 'rb'))
    features = pickle.load(open('feature_names.pkl', 'rb'))
    return model, scaler, encoders, features

model, scaler, encoders, features = load_models()

# --- 5. UI HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🚦 Location & Time-Based Traffic Congestion Prediction</h1>
    <p style="font-size: 1.2rem;">📍 <b>Visakhapatnam</b> — Smart City Traffic Analytics</p>
</div>
""", unsafe_allow_html=True)

# --- 6. INPUT SECTION ---
st.subheader("📍 Select Location, Date & Time")
col1, col2, col3 = st.columns(3)

with col1:
    location = st.selectbox("Location", ["Gajuwaka", "NAD Junction", "Maddilapalem", "Siripuram", 
                                         "MVP Colony", "Dwaraka Nagar", "RTC Complex", 
                                         "Jagadamba Junction", "Akkayyapalem", "Madhurawada"])
with col2:
    date = st.date_input("Date", datetime.now())
    day = date.strftime('%A')
    st.caption(f"📅 Detected Day: **{day}**")
with col3:
    st.write("⏰ Select Time")
    time_col1, time_col2, time_col3 = st.columns(3)
    with time_col1: hour_12 = st.selectbox("Hour", list(range(1, 13)), index=8)
    with time_col2: minute = st.selectbox("Min", [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], index=0)
    with time_col3: am_pm = st.radio("Period", ["AM", "PM"], index=0)
    if am_pm == "PM" and hour_12 != 12: hour = hour_12 + 12
    elif am_pm == "AM" and hour_12 == 12: hour = 0
    else: hour = hour_12
    st.caption(f"✅ Selected Time: {hour_12:02d}:{minute:02d} {am_pm}")
    time_display = f"{hour_12:02d}:{minute:02d} {am_pm}"

st.subheader("🌦️ Weather & Historical Traffic")
col4, col5, col6, col7 = st.columns(4)
with col4: temperature = st.slider("🌡️ Temperature (°C)", 20, 40, 30)
with col5: rainfall = st.slider("🌧️ Rainfall (mm)", 0.0, 50.0, 5.0, step=0.5)
with col6: prev_traffic = st.number_input("🚗 Traffic (1 hour ago)", min_value=50, max_value=1200, value=400)
with col7: prev_hour_traffic = st.number_input("🔄 Traffic (2 hours ago)", min_value=50, max_value=1200, value=380)

is_holiday = st.checkbox("🏖️ Is today a Public Holiday?")
st.markdown("---")

# --- 7. PREDICTION ---
if st.button("🔮 Predict Traffic Now", use_container_width=True, type="primary"):
    with st.spinner("🔍 Analyzing real-time Vizag traffic patterns..."):
        loc_encoded = encoders['Location'].transform([location])[0]
        day_encoded = encoders['Day'].transform([day])[0]
        is_weekend = 1 if day in ['Saturday', 'Sunday'] else 0
        holiday_flag = 1 if is_holiday else 0

        input_data = np.array([[loc_encoded, hour, day_encoded, is_weekend, holiday_flag,
                                temperature, rainfall, prev_traffic, prev_hour_traffic]])
        input_scaled = scaler.transform(input_data)
        pred_count = int(model.predict(input_scaled)[0])
        pred_count = max(20, pred_count)

        if pred_count > 700:
            level, traffic_text, congestion, color_class, speed = "🔴 HIGH", "High", "Severe Delays (>30 min)", "high-risk", np.random.normal(22, 5)
        elif pred_count > 400:
            level, traffic_text, congestion, color_class, speed = "🟡 MEDIUM", "Medium", "Moderate Slowdown (15-30 min)", "medium-risk", np.random.normal(38, 7)
        else:
            level, traffic_text, congestion, color_class, speed = "🟢 LOW", "Low", "Smooth Flow (<15 min)", "low-risk", np.random.normal(52, 8)
        speed = max(5, round(speed - rainfall * 0.3, 1))

        st.markdown("---")
        st.subheader("📊 Prediction Results")
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("🚗 Predicted Vehicle Count", f"{pred_count} vehicles")
        with m2: st.metric("📏 Estimated Avg. Speed", f"{speed} km/h")
        with m3:
            st.markdown(f"""
            <div class="traffic-card {color_class}">
                <div style="font-size: 2.2rem;">{level}</div>
                <div style="font-size: 1rem; font-weight: bold;">Traffic Level</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: #e9ecef; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <b>⏳ Expected Congestion:</b> {congestion}
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📋 Detailed Location Report")
        result_df = pd.DataFrame([{
            "City": "Visakhapatnam", "Location": location, "Date": date.strftime('%d-%m-%Y'),
            "Time": time_display, "Day": day, "Vehicle count": pred_count,
            "Average speed": f"{speed} km/h", "Traffic level": traffic_text,
            "Temperature": f"{temperature}°C", "Rainfall": f"{rainfall} mm",
            "Holiday": "Yes" if is_holiday else "No",
            "Traffic (1 hour ago)": prev_traffic, "Traffic (2 hours ago)": prev_hour_traffic
        }])
        
        # --- FIX: No matplotlib needed! Uses native BarColumn ---
        st.dataframe(
            result_df,
            column_config={
                "Vehicle count": st.column_config.BarColumn(
                    "Vehicle Count",
                    min_value=0,
                    max_value=1000,
                    width="medium",
                )
            },
            use_container_width=True,
            hide_index=True
        )

        # --- Plotly Gauge ---
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pred_count, domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "🚦 Traffic Volume Gauge"},
            gauge={'axis': {'range': [0, 1000]}, 'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 400], 'color': "lightgreen"},
                             {'range': [400, 700], 'color': "orange"},
                             {'range': [700, 1000], 'color': "salmon"}],
                   'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': pred_count}}
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👆 Adjust the inputs above and click **Predict** to see the traffic forecast!")
