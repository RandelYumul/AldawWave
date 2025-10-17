# train_model.py

import os
import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import warnings
from dotenv import load_dotenv
import joblib # Import joblib for model saving

warnings.filterwarnings("ignore")
load_dotenv()

# --- Configuration ---
csv_file = "angelesdataset.csv"
MODEL_FILE = 'aldaw_wave_model.joblib'
print(f"--- Aldaw-Wave Model Training Script ---")

# ==================================================
# STEP 1: Setup and Data Fetch (Identical to your existing logic)
# ==================================================
try:
    existing_df = pd.read_csv(csv_file)
    existing_df["time"] = pd.to_datetime(existing_df["time"], errors="coerce")
    existing_df = existing_df.dropna(subset=["time"])
    last_date = existing_df["time"].max()
    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"📅 Last date in dataset: {last_date.date()}, fetching from {start_date}")
    existing_df["time"] = pd.to_datetime(existing_df["time"]) 
except FileNotFoundError:
    print("🚀 No existing CSV found, starting fresh from 2023-01-01")
    existing_df = pd.DataFrame()
    start_date = "2023-01-01"

end_date = datetime.now().strftime("%Y-%m-%d")

if datetime.strptime(start_date, "%Y-%m-%d") >= datetime.strptime(end_date, "%Y-%m-%d"):
    print("✅ Dataset already up-to-date (no new archive data available).")
    data = existing_df.copy()
else:
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
         "latitude": 15.15, "longitude": 120.5833, "start_date": start_date, "end_date": end_date,
         "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "rain", "wind_speed_10m"],
         "timezone": "Asia/Singapore",
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    hourly_data = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m (°C)": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m (%)": hourly.Variables(1).ValuesAsNumpy(),
        "apparent_temperature (°C)": hourly.Variables(2).ValuesAsNumpy(),
        "rain (mm)": hourly.Variables(3).ValuesAsNumpy(),
        "wind_speed_10m (km/h)": hourly.Variables(4).ValuesAsNumpy(),
    }

    new_data_df = pd.DataFrame(hourly_data)
    updated_df = pd.concat([existing_df, new_data_df], ignore_index=True).drop_duplicates(subset=["time"])
    updated_df["time"] = pd.to_datetime(updated_df["time"], utc=True).dt.tz_convert(None)
    updated_df_save = updated_df.copy()
    updated_df_save["time"] = updated_df_save["time"].dt.strftime("%#m/%#d/%Y %I:%M:%S %p")
    for col in updated_df_save.columns:
         if updated_df_save[col].dtype in ["float64", "float32"]:
             updated_df_save[col] = updated_df_save[col].round(1)
    updated_df_save.to_csv(csv_file, index=False)
    print(f"✅ Data updated! Saved to {csv_file}")
    data = updated_df.copy()

if data is None or data.empty:
    print("❌ Fatal: Data could not be loaded or fetched. Training aborted.")
    exit()

# ==================================================
# STEP 2 & 3: Compute Heat Index, Feature Engineering, and Training
# ==================================================
def compute_heat_index(temp_c, humidity):
    temp_f = temp_c * 9/5 + 32
    hi_f = (-42.379 + 2.04901523*temp_f + 10.14333127*humidity
             - 0.22475541*temp_f*humidity - 0.00683783*temp_f**2
             - 0.05481717*humidity**2 + 0.00122874*temp_f**2*humidity
             + 0.00085282*temp_f*humidity**2 - 0.00000199*temp_f**2*humidity**2)
    return (hi_f - 32) * 5/9

data['heat_index'] = data.apply(
    lambda x: compute_heat_index(x['temperature_2m (°C)'], x['relative_humidity_2m (%)']),
    axis=1
)
data['hour'] = data['time'].dt.hour
data['day'] = data['time'].dt.day
data['month'] = data['time'].dt.month
data['is_weekend'] = data['time'].dt.dayofweek >= 5
data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)

X = data[['temperature_2m (°C)', 'relative_humidity_2m (%)',
          'rain (mm)', 'wind_speed_10m (km/h)',
          'hour', 'day', 'month', 'is_weekend', 'sin_hour', 'cos_hour']]
y = data['heat_index']

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [150, 200, 300], 'max_depth': [15, 20, None],
    'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]
}

print("\n⏳ Starting RandomizedSearchCV (Model Training)... This may take a moment.")
search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42), param_grid, n_iter=5, cv=3, scoring='r2', random_state=42
)
search.fit(X_train, y_train)
model = search.best_estimator_

# ==================================================
# STEP 4: Save the Trained Model
# ==================================================
joblib.dump(model, MODEL_FILE)
print(f"\n✨ Model training complete and saved as: {MODEL_FILE}")