# train_model.py - Part A: Load Data and Prep Features
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# 1. Load the data we just created
print("📂 Loading vizag_traffic.csv...")
df = pd.read_csv('vizag_traffic.csv')
print(f"✅ Loaded {len(df)} rows.")

# 2. Define our Features (Inputs) and Target (What we want to predict)
features = ['Location', 'Hour', 'Day', 'Is_Weekend', 'Is_Holiday', 
            'Temperature', 'Rainfall_mm', 'Previous_Traffic', 'Previous_Hour_Traffic']
target = 'Vehicle_Count'

print(f"\n🔍 Features: {features}")
print(f"🎯 Target: {target}")

# 3. Encode Categorical data (Text -> Numbers)
#    Machine Learning models can't read "Gajuwaka" or "Monday", so we turn them into 0, 1, 2...
from sklearn.preprocessing import LabelEncoder
label_encoders = {}

for col in ['Location', 'Day']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"✅ Encoded '{col}' into numbers.")

# 4. Split Data into X (inputs) and y (target)
X = df[features]
y = df[target]

print(f"\n📊 Input shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")

# 5. Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n✂️ Training set: {len(X_train)} rows")
# --- Part B: Train, Evaluate, and Save the Model ---

# 1. Scale Numerical Features (so the model treats 30°C and 500 cars fairly)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled successfully!")

# 2. Train the Random Forest Model
from sklearn.ensemble import RandomForestRegressor

print("\n🧠 Training the Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)
print("✅ Model training complete!")

# 3. Evaluate the Model on the Test Data
from sklearn.metrics import r2_score, mean_squared_error

y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 Model Performance:")
print(f"   - R² Score: {r2:.4f} (1.0 is perfect)")
print(f"   - RMSE: {rmse:.2f} vehicles (Lower is better)")

# 4. Save everything for the Web App to use
print("\n💾 Saving model and encoders...")
pickle.dump(model, open('traffic_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(label_encoders, open('label_encoders.pkl', 'wb'))
pickle.dump(features, open('feature_names.pkl', 'wb'))
print("✅ All files saved successfully!")

print("\n🎉 Phase 2 Complete! You now have a trained model ready for the app.")
