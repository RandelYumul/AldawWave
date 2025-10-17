# aldaw_wave_api.py

import os
import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import googlemaps
import google.generativeai as genai
import warnings
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import joblib

warnings.filterwarnings("ignore")
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
csv_file = "angelesdataset.csv"

MODEL_FILE = 'aldaw_wave_model.joblib'
app = Flask(__name__)
CORS(app)

# --- Global Variables for Pre-loaded Data/Model ---
data = None
model = None
hourly_avg = None # Pre-calculated for speed

# ==================================================
# Model Initialization Function (Run only once on startup)
# ==================================================

def initialize_model():
    """Handles data fetching, preprocessing, and model training."""
    global data, model, hourly_avg
    print("--- Initializing Aldaw-Wave Model ---")

    # STEP 1: Setup and Data Fetch
    try:
        existing_df = pd.read_csv(csv_file)
        existing_df["time"] = pd.to_datetime(existing_df["time"], errors="coerce")
        existing_df = existing_df.dropna(subset=["time"])
        last_date = existing_df["time"].max()
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"📅 Last date: {last_date.date()}, fetching from {start_date}")
        existing_df["time"] = pd.to_datetime(existing_df["time"]) # Convert back to datetime
    except FileNotFoundError:
        print("🚀 No existing CSV, starting fresh.")
        existing_df = pd.DataFrame()
        start_date = "2023-01-01"

    end_date = datetime.now().strftime("%Y-%m-%d")

    if datetime.strptime(start_date, "%Y-%m-%d") >= datetime.strptime(end_date, "%Y-%m-%d"):
        print("✅ Dataset already up-to-date (no new archive data available).")
        data = existing_df.copy()
    else:
        # OpenMeteo API call (kept concise for space)
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
        # Simplify saving format for robustness, keep original datetime for in-memory
        updated_df_save = updated_df.copy()
        updated_df_save["time"] = updated_df_save["time"].dt.strftime("%#m/%#d/%Y %I:%M:%S %p")
        for col in updated_df_save.columns:
             if updated_df_save[col].dtype in ["float64", "float32"]:
                 updated_df_save[col] = updated_df_save[col].round(1)
        updated_df_save.to_csv(csv_file, index=False)
        print(f"✅ Data updated! Saved to {csv_file}")
        data = updated_df.copy()

    if data is None or data.empty:
        print("❌ Fatal: Data could not be loaded or fetched.")
        return

    # STEP 2: Compute Heat Index
    def compute_heat_index(temp_c, humidity):
         temp_f = temp_c * 9/5 + 32
         # ... (heat index calculation formula from original script)
         hi_f = (-42.379 + 2.04901523*temp_f + 10.14333127*humidity
                  - 0.22475541*temp_f*humidity - 0.00683783*temp_f**2
                  - 0.05481717*humidity**2 + 0.00122874*temp_f**2*humidity
                  + 0.00085282*temp_f*humidity**2 - 0.00000199*temp_f**2*humidity**2)
         return (hi_f - 32) * 5/9

    data['heat_index'] = data.apply(
        lambda x: compute_heat_index(x['temperature_2m (°C)'], x['relative_humidity_2m (%)']),
        axis=1
    )

    # --- MODIFIED STEP 3: Load Model ---
    try:
        model = joblib.load(MODEL_FILE)
        print("✅ Pre-trained model loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Fatal: '{MODEL_FILE}' not found. Please run 'python train_model.py' first.")
        return # Stop initialization if model is missing

    # Pre-calculate hourly averages for prediction
    recent_days = data['time'].dt.date.unique()[-7:]
    recent_data = data[data['time'].dt.date.isin(recent_days)]
    hourly_avg = recent_data.groupby(recent_data['time'].dt.hour)[
        ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)', 'wind_speed_10m (km/h)']
    ].mean().reset_index().rename(columns={'time': 'hour'})

# ==================================================
# API Endpoint
# ==================================================

@app.route('/recommendation', methods=['POST'])
def get_recommendation():
    """API endpoint to run the model logic and return results."""
    if model is None or hourly_avg is None:
        return jsonify({"error": "Model not yet initialized. Please wait or check server logs."}), 503

    try:
        req_data = request.get_json()
        origin = req_data['origin']
        destination = req_data['destination']
        date_input = req_data['date']
        time_input = req_data['time']
    except Exception as e:
        return jsonify({"error": f"Invalid input data: {e}"}), 400

    try:
        arrival_dt = datetime.strptime(f"{date_input} {time_input}", "%Y-%m-%d %H:%M")
    except ValueError:
        return jsonify({"error": "Invalid date or time format. Use YYYY-MM-DD and HH:MM."}), 400

    # STEP 5: Google Maps Travel Time
    gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
    try:
        directions = gmaps.directions(origin, destination, mode="driving", departure_time="now")
        if not directions:
            travel_time = timedelta(minutes=30) # Default to 30 mins if no route
            travel_time_str = "30 minutes (fallback)"
        else:
            travel_seconds = directions[0]["legs"][0]["duration"]["value"]
            travel_time = timedelta(seconds=travel_seconds)
            travel_time_str = str(travel_time).split('.')[0]
    except Exception as e:
        print(f"❌ Error getting directions: {e}")
        travel_time = timedelta(minutes=30) # fallback
        travel_time_str = "30 minutes (fallback due to error)"

    # STEP 5B: Compute suggested departure times
    latest_departure = arrival_dt - travel_time
    suggested_departures = []
    # Try to get 3 times: on-time, 30 min early, 1 hour early
    intervals = [timedelta(minutes=60), timedelta(minutes=30), timedelta(minutes=0)]
    for interval in intervals:
        dep_time = latest_departure - interval
        if dep_time.date() == arrival_dt.date() and dep_time < arrival_dt:
             suggested_departures.append(dep_time)
        if len(suggested_departures) >= 3:
             break # Ensure we don't go back too far if travel time is short

    # STEP 6 & 7: Predict Heat Index and Get Gemini Recommendations
    departure_predictions = []
    gemini_recommendations = []
    
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel("gemini-2.0-flash-lite")

    for dep in suggested_departures:
        h = dep.hour
        if h in hourly_avg['hour'].values:
            base_data = hourly_avg[hourly_avg['hour']==h].iloc[0]
            features = pd.DataFrame([{
                 'temperature_2m (°C)': base_data['temperature_2m (°C)'],
                 'relative_humidity_2m (%)': base_data['relative_humidity_2m (%)'],
                 'rain (mm)': base_data['rain (mm)'],
                 'wind_speed_10m (km/h)': base_data['wind_speed_10m (km/h)'],
                 'hour': h, 'day': dep.day, 'month': dep.month,
                 'is_weekend': dep.weekday() >= 5,
                 'sin_hour': np.sin(2*np.pi*h/24),
                 'cos_hour': np.cos(2*np.pi*h/24)
            }])
            hi_pred = model.predict(features)[0]
            departure_predictions.append((dep, hi_pred))

            # Gemini Prompt
            prompt = f"""You are a travel assistant. Based on the following details, give a short, focused travel recommendation:
            - Departure time: {dep.strftime('%H:%M')}
            - Arrival time: {arrival_dt.strftime('%H:%M')}
            - Travel time: {travel_time_str}
            - Heat index: {hi_pred:.2f}°C
            - Origin: {origin}, Destination: {destination}, Date: {arrival_dt.strftime('%Y-%m-%d')}
            Provide only relevant advice and tips specific to this departure time, considering weather, heat index, and travel time. Keep the tone friendly and concise. Limit your response to 1-2 sentences only.
            """
            response = model_gemini.generate_content(prompt)
            gemini_recommendations.append({
                "departure_time": dep.strftime('%H:%M'),
                "heat_index": f"{hi_pred:.2f}°C",
                "recommendation": response.text.strip()
            })


    # STEP 8: Visualization
    hours_list = []
    heat_list = []
    for h in range(24):
        if h in hourly_avg['hour'].values:
            base_data = hourly_avg[hourly_avg['hour']==h].iloc[0]
            features = pd.DataFrame([{
                 'temperature_2m (°C)': base_data['temperature_2m (°C)'], 'relative_humidity_2m (%)': base_data['relative_humidity_2m (%)'],
                 'rain (mm)': base_data['rain (mm)'], 'wind_speed_10m (km/h)': base_data['wind_speed_10m (km/h)'],
                 'hour': h, 'day': arrival_dt.day, 'month': arrival_dt.month,
                 'is_weekend': arrival_dt.weekday() >= 5, 'sin_hour': np.sin(2*np.pi*h/24), 'cos_hour': np.cos(2*np.pi*h/24)
            }])
            hi_pred = model.predict(features)[0]
            hours_list.append(h)
            heat_list.append(hi_pred)

    plt.figure(figsize=(12, 5))
    plt.plot(hours_list, heat_list, marker='o', color='orange', label='Hourly Predicted Heat Index')

    for dep, hi in departure_predictions:
        dep_hour_float = dep.hour + dep.minute / 60
        plt.scatter(dep_hour_float, hi, color='red', s=90, zorder=5)
        plt.text(dep_hour_float + 0.05, hi + 0.25, f"{dep.strftime('%H:%M')}", color='red', fontsize=9, weight='bold')

    plt.xlabel("Hour of Day")
    plt.ylabel("Heat Index (°C)")
    plt.title(f"Hourly Heat Index Forecast ({arrival_dt.strftime('%Y-%m-%d')})\n{origin} ➜ {destination}")
    plt.xticks(range(0, 24))
    plt.xlim(-0.5, 23.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save plot to an in-memory buffer, encode to base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close() # Close plot figure to free memory
    graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Prepare final JSON response
    return jsonify({
        "graph_image": f"data:image/png;base64,{graph_base64}",
        "recommendations": gemini_recommendations,
        "travel_time": travel_time_str
    })


if __name__ == '__main__':
    initialize_model() # Initialize model on server start
    # The server will run on http://127.0.0.1:5000/
    app.run(debug=True, use_reloader=False)