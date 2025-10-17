import os
import openmeteo_requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests_cache
from retry_requests import retry
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import googlemaps
import google.generativeai as genai
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

load_dotenv()  # Load .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==================================================
# STEP 1: Setup and Data Fetch
# ==================================================
csv_file = "angelesdataset.csv"

try:
    existing_df = pd.read_csv(csv_file)
    existing_df["time"] = pd.to_datetime(existing_df["time"], errors="coerce")
    existing_df = existing_df.dropna(subset=["time"])

    last_date = existing_df["time"].max()
    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"📅 Last date in dataset: {last_date.date()}, fetching from {start_date}")
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
        "latitude": 15.15,
        "longitude": 120.5833,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m", "relative_humidity_2m",
            "apparent_temperature", "rain", "wind_speed_10m"
        ],
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
    updated_df["time"] = updated_df["time"].dt.strftime("%#m/%#d/%Y %I:%M:%S %p")
    for col in updated_df.columns:
        if updated_df[col].dtype in ["float64", "float32"]:
            updated_df[col] = updated_df[col].round(1)
    updated_df.to_csv(csv_file, index=False)
    print(f"✅ Data updated! Saved to {csv_file}")
    data = updated_df.copy()

if 'data' not in locals():
    data = existing_df.copy()

# ==================================================
# STEP 2: Compute Heat Index
# ==================================================
data['time'] = pd.to_datetime(data['time'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
data = data.dropna(subset=['time'])

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

# ==================================================
# STEP 3: Feature Engineering & Model Training
# ==================================================
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [150, 200, 300],
    'max_depth': [15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid, n_iter=5, cv=3, scoring='r2', random_state=42
)
search.fit(X_train, y_train)
model = search.best_estimator_
print("✅ Model training complete!")

# ==================================================
# STEP 4: User Input
# ==================================================
origin = input("Enter origin location: ")
destination = input("Enter destination location: ")
date_input = input("Enter arrival date (YYYY-MM-DD): ")
time_input = input("Enter arrival time (HH:MM, 24-hour): ")

arrival_dt = datetime.strptime(f"{date_input} {time_input}", "%Y-%m-%d %H:%M")

# ==================================================
# STEP 5: Google Maps Travel Time
# ==================================================
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

try:
    directions = gmaps.directions(
        origin,
        destination,
        mode="driving",
        departure_time="now"
    )

    if not directions:
        print("⚠️ No route found. Please check the location names or your API key permissions.")
        travel_time = timedelta(minutes=0)  # fallback
    else:
        travel_seconds = directions[0]["legs"][0]["duration"]["value"]
        travel_time = timedelta(seconds=travel_seconds)
        print(f"🚗 Estimated travel time: {travel_time}")

except Exception as e:
    print(f"❌ Error getting directions: {e}")
    travel_time = timedelta(minutes=0)  # fallback


print(f"\n📍 Route: {origin} ➜ {destination}")
print(f"🎯 Desired arrival: {arrival_dt.strftime('%Y-%m-%d %H:%M')}")
print(f"🚗 Estimated travel time: {travel_time}")

# ==================================================
# STEP 5B: Compute up to 3 valid pre-arrival departure times
# ==================================================
# Calculate the latest possible departure (to arrive exactly on time)
latest_departure = arrival_dt - travel_time

# Step backwards in 30-minute intervals, but don’t go past midnight or before 6 AM
suggested_departures = []
for i in range(2, -1, -1):  # gives 2, 1, 0
    dep_time = latest_departure - timedelta(minutes=30 * i)
    if dep_time.date() == arrival_dt.date() and dep_time.time() >= datetime.strptime("06:00", "%H:%M").time():
        suggested_departures.append(dep_time)

# ==================================================
# STEP 6: Predict Heat Index for Each Departure
# ==================================================
recent_days = data['time'].dt.date.unique()[-7:]
recent_data = data[data['time'].dt.date.isin(recent_days)]
hourly_avg = recent_data.groupby(recent_data['time'].dt.hour)[
    ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)', 'wind_speed_10m (km/h)']
].mean().reset_index().rename(columns={'time': 'hour'})

departure_predictions = []
for dep in suggested_departures:
    h = dep.hour
    if h in hourly_avg['hour'].values:
        base_temp = hourly_avg.loc[hourly_avg['hour']==h, 'temperature_2m (°C)'].values[0]
        base_humid = hourly_avg.loc[hourly_avg['hour']==h, 'relative_humidity_2m (%)'].values[0]
        base_rain = hourly_avg.loc[hourly_avg['hour']==h, 'rain (mm)'].values[0]
        base_wind = hourly_avg.loc[hourly_avg['hour']==h, 'wind_speed_10m (km/h)'].values[0]

        features = pd.DataFrame([{
            'temperature_2m (°C)': base_temp,
            'relative_humidity_2m (%)': base_humid,
            'rain (mm)': base_rain,
            'wind_speed_10m (km/h)': base_wind,
            'hour': h, 'day': dep.day, 'month': dep.month,
            'is_weekend': dep.weekday() >= 5,
            'sin_hour': np.sin(2*np.pi*h/24),
            'cos_hour': np.cos(2*np.pi*h/24)
        }])
        hi_pred = model.predict(features)[0]
        departure_predictions.append((dep, hi_pred))

departure_predictions = [
    (dep_time, hi)
    for dep_time, hi in departure_predictions
    if dep_time <= arrival_dt
]

print("\n🕒 Recommended Departures:")
for i, (dep, hi) in enumerate(departure_predictions, start=1):
    print(f"{i}. {dep.strftime('%H:%M')} — Predicted Heat Index: {hi:.2f}°C")

# ==================================================
# STEP 7: Gemini Recommendations
# ==================================================
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.0-flash-lite")

print("\n💡 Gemini Travel Tips:\n")

for i, (departure, heat_index) in enumerate(departure_predictions, start=1):
    prompt = f"""
    You are a travel assistant. Based on the following details, give a short, focused travel recommendation:

    - Departure time: {departure.strftime('%H:%M')}
    - Arrival time: {arrival_dt.strftime('%H:%M')}
    - Travel time: {travel_time}
    - Heat index: {heat_index:.2f}°C
    - Origin: {origin}
    - Destination: {destination}
    - Date: {arrival_dt.strftime('%Y-%m-%d')}

    Provide only relevant advice and tips specific to this departure time,
    considering weather, heat index, and travel time.
    Keep the tone friendly and concise.
    Limit your response to 1-2 sentences only.
    """

    response = model_gemini.generate_content(prompt)
    print(f"🕒 Recommendation {i} ({departure.strftime('%H:%M')}):")
    print(response.text.strip(), "\n")

# ==================================================
# STEP 8: Visualization (Hourly + Recommended Departures)
# ==================================================
plt.figure(figsize=(11, 5))

# ---- Predict hourly heat index for the entire input day ----
hours = range(6, 19)  # 6 AM to 6 PM
hourly_predictions = []
for h in hours:
    if h in hourly_avg['hour'].values:
        base_temp = hourly_avg.loc[hourly_avg['hour']==h, 'temperature_2m (°C)'].values[0]
        base_humid = hourly_avg.loc[hourly_avg['hour']==h, 'relative_humidity_2m (%)'].values[0]
        base_rain = hourly_avg.loc[hourly_avg['hour']==h, 'rain (mm)'].values[0]
        base_wind = hourly_avg.loc[hourly_avg['hour']==h, 'wind_speed_10m (km/h)'].values[0]

        features = pd.DataFrame([{
            'temperature_2m (°C)': base_temp,
            'relative_humidity_2m (%)': base_humid,
            'rain (mm)': base_rain,
            'wind_speed_10m (km/h)': base_wind,
            'hour': h, 'day': arrival_dt.day, 'month': arrival_dt.month,
            'is_weekend': arrival_dt.weekday() >= 5,
            'sin_hour': np.sin(2*np.pi*h/24),
            'cos_hour': np.cos(2*np.pi*h/24)
        }])
        hi_pred = model.predict(features)[0]
        hourly_predictions.append((h, hi_pred))

# Convert to arrays
hours_list = [h for h, _ in hourly_predictions]
heat_list = [hi for _, hi in hourly_predictions]

# ---- Plot hourly prediction curve ----
plt.plot(hours_list, heat_list, marker='o', color='orange', label='Hourly Predicted Heat Index')

# ---- Mark and label the recommended departures ----
for i, (dep, hi) in enumerate(departure_predictions, start=1):
    # Convert time to fractional hour (e.g. 14:30 → 14.5)
    dep_hour_float = dep.hour + dep.minute / 60
    plt.scatter(dep_hour_float, hi, color='red', s=90, zorder=5)
    plt.text(dep_hour_float + 0.05, hi + 0.25,
             f"{dep.strftime('%H:%M')}",
             color='red', fontsize=9, weight='bold')

# ---- Formatting ----
plt.xlabel("Hour of Day")
plt.ylabel("Heat Index (°C)")
plt.title(f"Hourly Heat Index Forecast ({arrival_dt.strftime('%Y-%m-%d')})\n{origin} ➜ {destination}")

# X-axis ticks every hour
plt.xticks(range(6, 19))
plt.xlim(5.5, 18.5)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


