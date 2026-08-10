# app.py - Complete Vizag Traffic Predictor (With AM/PM Dropdowns)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.graph_objects as go
import os
import subprocess

# --- AUTO-BUILD MODELS IF MISSING (FOR STREAMLIT CLOUD) ---
if not os.path.exists('traffic_model.pkl'):
    st.warning("⚙️ First time setup! Training the Vizag model on the cloud... (~60 seconds)")
    subprocess.run(['python', 'train_model.py'], check=True)
    st.success("✅ Model ready! Refreshing...")
    st.rerun()
# --- Page Configuration ---
st.set_page_config(
    page_title="🚦 Vizag Traffic Predictor",
    page_icon="🚦",
    layout="wide"
)

# --- Custom CSS for Aesthetic UI ---
st.markdown("""
<style>
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
    .high-risk {
        background-color: #ffcccc;
        border-left: 8px solid #dc3545;
    }
    .medium-risk {
        background-color: #ffe5b4;
        border-left: 8px solid #fd7e14;
    }
    .low-risk {
        background-color: #d4edda;
        border-left: 8px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# --- Load the saved models ---
@st.cache_resource
def load_models():
    model = pickle.load(open('traffic_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    encoders = pickle.load(open('label_encoders.pkl', 'rb'))
    features = pickle.load(open('feature_names.pkl', 'rb'))
    return model, scaler, encoders, features

model, scaler, encoders, features = load_models()

# --- Page Header ---
st.markdown("""
<div class="main-header">
    <h1>🚦 Location & Time-Based Traffic Congestion Prediction</h1>
    <p style="font-size: 1.2rem;">📍 <b>Visakhapatnam</b> — Smart City Traffic Analytics</p>
</div>
""", unsafe_allow_html=True)

# --- INPUT SECTION: Row 1 (Location, Date, Time) ---
st.subheader("📍 Select Location, Date & Time")
col1, col2, col3 = st.columns(3)

with col1:
    location = st.selectbox(
        "Location", 
        ["Gajuwaka", "NAD Junction", "Maddilapalem", "Siripuram", 
         "MVP Colony", "Dwaraka Nagar", "RTC Complex", 
         "Jagadamba Junction", "Akkayyapalem", "Madhurawada"]
    )

with col2:
    date = st.date_input("Date", datetime.now())
    day = date.strftime('%A')
    st.caption(f"📅 Detected Day: **{day}**")

with col3:
    st.write("⏰ Select Time")
    
    # Custom AM/PM Dropdowns (Guarantees 12-hour format)
    time_col1, time_col2, time_col3 = st.columns(3)
    
    with time_col1:
        hour_12 = st.selectbox("Hour", list(range(1, 13)), index=8)  # Default 9
    with time_col2:
        minute = st.selectbox("Min", [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], index=0)
    with time_col3:
        am_pm = st.radio("Period", ["AM", "PM"], index=0)
    
    # Convert to 24-hour format for the ML model
    if am_pm == "PM" and hour_12 != 12:
        hour = hour_12 + 12
    elif am_pm == "AM" and hour_12 == 12:
        hour = 0
    else:
        hour = hour_12
    
    st.caption(f"✅ Selected Time: {hour_12:02d}:{minute:02d} {am_pm}")
    
    # Create a time object for output formatting
    time_display = f"{hour_12:02d}:{minute:02d} {am_pm}"

# --- INPUT SECTION: Row 2 (Weather & History) ---
st.subheader("🌦️ Weather & Historical Traffic")
col4, col5, col6, col7 = st.columns(4)  # <-- THIS LINE FIXES THE "col4" ERROR

with col4:
    temperature = st.slider("🌡️ Temperature (°C)", 20, 40, 30)

with col5:
    rainfall = st.slider("🌧️ Rainfall (mm)", 0.0, 50.0, 5.0, step=0.5)

with col6:
    prev_traffic = st.number_input("🚗 Traffic (1 hour ago)", min_value=50, max_value=1200, value=400)

with col7:
    prev_hour_traffic = st.number_input("🔄 Traffic (2 hours ago)", min_value=50, max_value=1200, value=380)

# --- Holiday Toggle ---
is_holiday = st.checkbox("🏖️ Is today a Public Holiday?")

# --- PREDICTION BUTTON ---
st.markdown("---")

if st.button("🔮 Predict Traffic Now", use_container_width=True, type="primary"):
    with st.spinner("🔍 Analyzing real-time Vizag traffic patterns..."):
        
        # Encode inputs
        loc_encoded = encoders['Location'].transform([location])[0]
        day_encoded = encoders['Day'].transform([day])[0]
        is_weekend = 1 if day in ['Saturday', 'Sunday'] else 0
        holiday_flag = 1 if is_holiday else 0

        # Prepare input array (9 features)
        input_data = np.array([[
            loc_encoded, hour, day_encoded, is_weekend, holiday_flag,
            temperature, rainfall, prev_traffic, prev_hour_traffic
        ]])
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        pred_count = int(model.predict(input_scaled)[0])
        pred_count = max(20, pred_count)

        # Determine Traffic Level & Speed
        if pred_count > 700:
            level = "🔴 HIGH"
            traffic_text = "High"
            congestion = "Severe Delays (>30 min)"
            color_class = "high-risk"
            speed = np.random.normal(22, 5)
        elif pred_count > 400:
            level = "🟡 MEDIUM"
            traffic_text = "Medium"
            congestion = "Moderate Slowdown (15-30 min)"
            color_class = "medium-risk"
            speed = np.random.normal(38, 7)
        else:
            level = "🟢 LOW"
            traffic_text = "Low"
            congestion = "Smooth Flow (<15 min)"
            color_class = "low-risk"
            speed = np.random.normal(52, 8)
        
        speed = max(5, round(speed - rainfall * 0.3, 1))

        # --- DISPLAY RESULTS ---
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("🚗 Predicted Vehicle Count", f"{pred_count} vehicles")
        with metric_col2:
            st.metric("📏 Estimated Avg. Speed", f"{speed} km/h")
        with metric_col3:
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
        
        # --- THE EXACT OUTPUT TABLE ---
        st.subheader("📋 Detailed Location Report")
        result_df = pd.DataFrame([{
            "City": "Visakhapatnam",
            "Location": location,
            "Date": date.strftime('%d-%m-%Y'),
            "Time": time_display,  # Shows 09:00 AM format
            "Day": day,
            "Vehicle count": pred_count,
            "Average speed": f"{speed} km/h",
            "Traffic level": traffic_text,
            "Temperature": f"{temperature}°C",
            "Rainfall": f"{rainfall} mm",
            "Holiday": "Yes" if is_holiday else "No",
            "Traffic (1 hour ago)": prev_traffic,
            "Traffic (2 hours ago)": prev_hour_traffic
        }])
        
        st.dataframe(
            result_df.style.background_gradient(cmap='Blues', subset=['Vehicle count']), 
            use_container_width=True,
            hide_index=True
        )
        
        # --- BONUS: Traffic Gauge Chart ---
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_count,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "🚦 Traffic Volume Gauge"},
            gauge={
                'axis': {'range': [0, 1000]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 400], 'color': "lightgreen"},
                    {'range': [400, 700], 'color': "orange"},
                    {'range': [700, 1000], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': pred_count
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Adjust the inputs above and click **Predict** to see the traffic forecast for Visakhapatnam!")
