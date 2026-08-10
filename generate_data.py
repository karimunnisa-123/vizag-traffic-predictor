import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Seed for reproducibility
np.random.seed(42)

# 1. Define Visakhapatnam's 10 key locations
# Each location has a "Base Traffic" volume (Industrial > Residential)
locations = {
    "Gajuwaka": 450,        # Industrial hub
    "NAD Junction": 420,    # Busy intersection
    "Maddilapalem": 380,    # Educational area
    "Siripuram": 400,       # Commercial
    "MVP Colony": 350,      # Residential
    "Dwaraka Nagar": 390,   # Shopping area
    "RTC Complex": 500,     # Busiest bus stand
    "Jagadamba Junction": 480, # Entertainment hub
    "Akkayyapalem": 370,    # Mixed area
    "Madhurawada": 280      # IT/Residential suburb
}

print("✅ Visakhapatnam locations loaded!")
# --- Part B: Generate the traffic data ---

# Create a list of all dates from Jan 1, 2024 to today (Aug 10, 2026)
start_date = datetime(2024, 1, 1)
end_date = datetime(2026, 8, 10)
date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

# This will hold all our rows of data
all_rows = []

# Loop through every single day
for single_date in date_list:
    # Loop through every location in Vizag
    for loc_name, base_traffic in locations.items():
        
        # Loop through hours from 6 AM to 10 PM (16 hours of traffic data)
        for hour in range(6, 23):
            
            # --- 1. Calculate Day Info ---
            day_name = single_date.strftime('%A')
            is_weekend = 1 if day_name in ['Saturday', 'Sunday'] else 0
            
            # --- 2. Check for Indian Holidays (Simplified) ---
            is_holiday = 0
            if single_date.month == 1 and single_date.day == 26: is_holiday = 1  # Republic Day
            if single_date.month == 8 and single_date.day == 15: is_holiday = 1  # Independence Day
            if single_date.month == 10 and single_date.day == 2: is_holiday = 1   # Gandhi Jayanti
            if single_date.month == 12 and single_date.day == 25: is_holiday = 1  # Christmas
            # Add a small random chance for local Vizag festivals
            if random.random() < 0.02: is_holiday = 1 

            # --- 3. Apply "Time Multipliers" (When do people drive?) ---
            if 8 <= hour <= 10:   # Morning Office/School Rush
                time_mult = 2.2
            elif 17 <= hour <= 19: # Evening Rush (5 PM - 7 PM)
                time_mult = 2.5
            elif 12 <= hour <= 14: # Lunch time
                time_mult = 1.3
            elif 21 <= hour <= 22: # Late night
                time_mult = 0.6
            else:
                time_mult = 0.9   # Normal midday / early morning

            # If it's a weekend or holiday, people drive more during the day for shopping
            if is_weekend or is_holiday:
                if 10 <= hour <= 20:  
                    time_mult = 1.4  # Shopping rush
                else:
                    time_mult = 0.7  # People sleep in / stay home

            # --- 4. Vizag Weather (Tropical) ---
            temperature = round(np.random.normal(30, 4), 1)  # Avg 30°C
            
            # Rainy season in Vizag (June to September)
            if single_date.month in [6, 7, 8, 9]:
                rainfall = round(max(0, np.random.exponential(8)), 1)
            else:
                rainfall = round(max(0, np.random.exponential(2)), 1)
            
            # Heavy rain reduces the number of vehicles on the road
            if rainfall > 15:
                time_mult = time_mult * 0.8  

            # --- 5. Calculate the Final Vehicle Count (Our Target) ---
            # Add some random variation to make it realistic
            random_noise = np.random.normal(0, 40)
            vehicle_count = int(max(20, base_traffic * time_mult + random_noise))
            
            # --- 6. Calculate Average Speed (Depends on traffic) ---
            if vehicle_count > 700:
                avg_speed = round(max(5, np.random.normal(22, 5)), 1)
            elif vehicle_count > 400:
                avg_speed = round(max(10, np.random.normal(38, 7)), 1)
            else:
                avg_speed = round(max(15, np.random.normal(52, 10)), 1)
            
            # Rain makes roads slower
            avg_speed = round(max(5, avg_speed - rainfall * 0.3), 1)

            # --- 7. Previous Traffic (Simulate history) ---
            previous_traffic = int(max(50, vehicle_count + np.random.normal(0, 30)))
            previous_hour_traffic = int(max(50, vehicle_count + np.random.normal(0, 20)))

            # Save this row of data
            all_rows.append({
                'City': 'Visakhapatnam',
                'Location': loc_name,
                'Date': single_date.strftime('%Y-%m-%d'),
                'Time': f"{hour:02d}:00",
                'Hour': hour,
                'Day': day_name,
                'Is_Weekend': is_weekend,
                'Is_Holiday': is_holiday,
                'Temperature': temperature,
                'Rainfall_mm': rainfall,
                'Previous_Traffic': previous_traffic,
                'Previous_Hour_Traffic': previous_hour_traffic,
                'Vehicle_Count': vehicle_count,
                'Avg_Speed_kmh': avg_speed
            })

# Convert to a Pandas DataFrame and save to CSV
df = pd.DataFrame(all_rows)
df.to_csv('vizag_traffic.csv', index=False)

print(f"✅ SUCCESS! Generated {len(df)} rows of Visakhapatnam traffic data!")
print("\n📊 Here are the first 5 rows of your dataset:")
print(df.head())
