import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import requests_cache
from retry_requests import retry
import openmeteo_requests

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score

from flask import Flask, request, jsonify
from flask_cors import CORS

import googlemaps
import google.generativeai as genai

import matplotlib.pyplot as plt
import base64
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")

# ---------- Configuration ----------
CSV_FILE = "angelesdataset.csv"
MODEL_FILE = "aldaw_wave_model.joblib"
MODEL_META_FILE = "aldaw_wave_model_meta.joblib"  # store metadata: last_trained_on (datetime)
CACHE_DB = ".cache"
OPENMETEO_LAT = 15.15
OPENMETEO_LON = 120.5833
TIMEZONE = "Asia/Singapore"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Training control parameters
RANDOM_SEARCH_ITERS = int(os.getenv("RANDOM_SEARCH_ITERS", "5"))  # keep small for speed
CV_FOLDS = int(os.getenv("CV_FOLDS", "3"))
R2_DOWNGRADE_THRESHOLD = float(os.getenv("R2_DOWNGRADE_THRESHOLD", "0.02"))  # retrain if r2 drop > this
MODEL_MIN_TRAIN_ROWS = int(os.getenv("MODEL_MIN_TRAIN_ROWS", "200"))  # avoid training on tiny datasets

# ---------- Globals ----------
app = Flask(__name__)
CORS(app)

data = None
model = None
model_meta = {}
hourly_avg = None

# ---------- Utility functions ----------
def compute_heat_index_vectorized(temp_c, humidity):
    # Vectorized heat-index (F-based polynomial) -> returns Celsius
    temp_f = temp_c * 9/5 + 32
    hi_f = (-42.379 + 2.04901523*temp_f + 10.14333127*humidity
            - 0.22475541*temp_f*humidity - 0.00683783*temp_f**2
            - 0.05481717*humidity**2 + 0.00122874*temp_f**2*humidity
            + 0.00085282*temp_f*humidity**2 - 0.00000199*temp_f**2*humidity**2)
    return (hi_f - 32) * 5/9

def load_csv_or_empty(csv_file=CSV_FILE):
    try:
        df = pd.read_csv(csv_file)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def fetch_openmeteo_archive(start_date, end_date):
    # returns a DataFrame
    cache_session = requests_cache.CachedSession(CACHE_DB, expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": OPENMETEO_LAT,
        "longitude": OPENMETEO_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "rain", "wind_speed_10m"],
        "timezone": TIMEZONE,
    }
    responses = client.weather_api(url, params=params)
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
    df_new = pd.DataFrame(hourly_data)
    df_new["time"] = pd.to_datetime(df_new["time"], utc=True).dt.tz_convert(None)
    return df_new

def prepare_features(df):
    # Input: DataFrame with required columns
    df = df.copy()
    df["heat_index"] = compute_heat_index_vectorized(df["temperature_2m (°C)"], df["relative_humidity_2m (%)"])
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["is_weekend"] = (df["time"].dt.dayofweek >= 5).astype(int)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

    X = df[['temperature_2m (°C)', 'relative_humidity_2m (%)',
            'rain (mm)', 'wind_speed_10m (km/h)',
            'hour', 'day', 'month', 'is_weekend', 'sin_hour', 'cos_hour']].copy()
    y = df['heat_index'].copy()
    return X, y, df

def save_all_predicted_heat_index(data_prepared, model, output_file="predicted_heat_index.csv"):
    """
    Predicts heat index for each timestamp in the dataset and saves results hourly.
    Columns: time, predicted_heat_index, date, hour
    """
    if model is None or data_prepared is None or data_prepared.empty:
        print("⚠️ Skipping prediction save: model or data not ready.")
        return

    feature_cols = ['temperature_2m (°C)', 'relative_humidity_2m (%)',
                    'rain (mm)', 'wind_speed_10m (km/h)',
                    'hour', 'day', 'month', 'is_weekend', 'sin_hour', 'cos_hour']
    
    X = data_prepared[feature_cols]
    preds = model.predict(X)

    df_preds = pd.DataFrame({
        "time": data_prepared["time"],
        "predicted_heat_index": preds
    })

    # Add date and hour columns for filtering later
    df_preds["date"] = df_preds["time"].dt.date
    df_preds["hour"] = df_preds["time"].dt.hour

    # Save hourly predictions directly
    df_preds.to_csv(output_file, index=False)
    print(f"Saved hourly predicted heat index to {output_file} ({len(df_preds)} rows)")


def quick_train_random_forest(X_train, y_train, n_iter=RANDOM_SEARCH_ITERS):
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [12, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    base = RandomForestRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(base, param_distributions=param_grid, n_iter=n_iter, cv=CV_FOLDS,
                                scoring='r2', random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    return best, search.best_score_

def save_model_and_meta(model_obj, meta_obj, model_file=MODEL_FILE, meta_file=MODEL_META_FILE):
    joblib.dump(model_obj, model_file)
    joblib.dump(meta_obj, meta_file)

def load_model_and_meta(model_file=MODEL_FILE, meta_file=MODEL_META_FILE):
    try:
        m = joblib.load(model_file)
        meta = joblib.load(meta_file) if os.path.exists(meta_file) else {}
        return m, meta
    except Exception:
        return None, {}

# ---------- Main initialization logic ----------
def initialize_model(force_retrain=False):
    """
    Loads CSV, fetches new archive data if available, updates CSV, prepares features.
    Loads the model if possible and decides whether to retrain.
    """
    global data, model, model_meta, hourly_avg
    print("--- Initializing Aldaw-Wave Combined Service ---")

    # Load existing CSV
    existing_df = load_csv_or_empty(CSV_FILE)
    if existing_df.empty:
        start_date = "2023-01-01"
        print("No existing CSV found; will fetch data starting 2023-01-01.")
    else:
        last_date = existing_df["time"].max()
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Last date in CSV: {last_date.date()}, fetch from: {start_date}")

    end_date = datetime.now().strftime("%Y-%m-%d")
    if datetime.strptime(start_date, "%Y-%m-%d") < datetime.strptime(end_date, "%Y-%m-%d"):
        # Fetch only if there's new archive data
        try:
            df_new = fetch_openmeteo_archive(start_date, end_date)
            if not df_new.empty:
                updated_df = pd.concat([existing_df, df_new], ignore_index=True).drop_duplicates(subset=["time"])
                updated_df["time"] = pd.to_datetime(updated_df["time"], utc=True).dt.tz_convert(None)
                # Save with format of the CSV
                updated_df_save = updated_df.copy()
                updated_df_save["time"] = updated_df_save["time"].dt.strftime("%#m/%#d/%Y %I:%M:%S %p")
                for col in updated_df_save.columns:
                    if updated_df_save[col].dtype in ["float64", "float32"]:
                        updated_df_save[col] = updated_df_save[col].round(1)
                updated_df_save.to_csv(CSV_FILE, index=False)
                print(f"Data updated and saved to {CSV_FILE} (new rows: {len(df_new)})")
                data = updated_df.copy()
            else:
                print("No new archived rows returned.")
                data = existing_df.copy()
        except Exception as e:
            print(f"Error fetching archive data: {e}. Using existing CSV if present.")
            data = existing_df.copy()
    else:
        print("Dataset already up-to-date.")
        data = existing_df.copy()

    if data is None or data.empty:
        raise RuntimeError("No data available for training or prediction.")

    # Prepare features
    X_all, y_all, data_prepared = prepare_features(data)

    # Load model and meta if present
    loaded_model, loaded_meta = load_model_and_meta()
    model = loaded_model
    model_meta = loaded_meta or {}

    # Decide whether to retrain:
    retrain_needed = force_retrain
    reason = None

    if model is None:
        retrain_needed = True
        reason = "no existing model"
    else:
        # Evaluate on recent slice to see if model still performs well
        try:
            recent_cutoff = data_prepared["time"].max() - pd.Timedelta(days=7)
            recent_mask = data_prepared["time"] >= recent_cutoff
            recent_df = data_prepared[recent_mask]
            if len(recent_df) >= 50:
                X_recent, y_recent, _ = prepare_features(recent_df)
                preds = model.predict(X_recent)
                current_r2 = r2_score(y_recent, preds)
                last_trained_r2 = model_meta.get("last_train_r2", None)
                print(f"Model present. Recent R2 on last 7 days: {current_r2:.4f} (stored r2: {last_trained_r2})")
                if last_trained_r2 is None:
                    # No stored metric; be conservative: retrain if current_r2 is low
                    if current_r2 < 0.4:  # arbitrary threshold
                        retrain_needed = True
                        reason = "low recent performance"
                else:
                    if (last_trained_r2 - current_r2) > R2_DOWNGRADE_THRESHOLD:
                        retrain_needed = True
                        reason = f"r2 dropped by {last_trained_r2 - current_r2:.4f}"
            else:
                # Not enough recent rows: skip retrain by default
                print("Not enough recent rows to evaluate performance; skipping automatic retrain check.")
        except Exception as e:
            print("Error evaluating existing model on recent data:", e)
            retrain_needed = True
            reason = "evaluation failed"

    # If retrain needed, reprocess (synchronously)
    if retrain_needed:
        if len(X_all) < MODEL_MIN_TRAIN_ROWS:
            # If dataset too small, still try to train
            print(f"WARNING: Only {len(X_all)} rows available. Training may be unreliable.")
        print(f"⏳ Retraining model because: {reason or 'forced'} ...")
        # Use a train-test split and quick randomized search
        X_train, X_hold, y_train, y_hold = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        new_model, best_score = quick_train_random_forest(X_train, y_train, n_iter=RANDOM_SEARCH_ITERS)
        # Evaluate on holdout
        hold_preds = new_model.predict(X_hold)
        hold_r2 = r2_score(y_hold, hold_preds)
        # Save model & meta
        model_meta = {
            "last_trained_on": datetime.utcnow(),
            "train_rows": len(X_all),
            "best_search_score": float(best_score),
            "last_train_r2": float(hold_r2)
        }
        save_model_and_meta(new_model, model_meta)
        model = new_model
        print(f"Retrained model. Holdout R2: {hold_r2:.4f}. Model saved to {MODEL_FILE}.")
    else:
        print("Using existing model without retraining.")

    # Precompute hourly averages for prediction speed
    recent_days = sorted(data_prepared['time'].dt.date.unique())[-7:]
    recent_data = data_prepared[data_prepared['time'].dt.date.isin(recent_days)]
    if not recent_data.empty:
        hourly_avg = recent_data.groupby(recent_data['time'].dt.hour)[
            ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)', 'wind_speed_10m (km/h)']
        ].mean().reset_index().rename(columns={'time': 'hour'})
    else:
        hourly_avg = data_prepared.groupby(data_prepared['time'].dt.hour)[
            ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)', 'wind_speed_10m (km/h)']
        ].mean().reset_index().rename(columns={'time': 'hour'})

    # attach to globals
    globals().update({"data": data_prepared, "model": model, "model_meta": model_meta, "hourly_avg": hourly_avg})
    print("Initialization complete.")
    save_all_predicted_heat_index(data_prepared, model)


# ---------- Flask endpoints ----------
@app.route("/get_api_key", methods=["GET"])
def get_api_key():
    return jsonify({"key": GOOGLE_API_KEY})


@app.route("/recommendation", methods=["POST"])
def get_recommendation():
    global model, hourly_avg
    if model is None or hourly_avg is None:
        return jsonify({"error": "Model not yet initialized. Please check server logs."}), 503

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

     # Try to load precomputed heat index for the given date
    predicted_file = "predicted_heat_index.csv"
    target_date = datetime.strptime(date_input, "%Y-%m-%d").date()
    hourly_for_date = None

    if os.path.exists(predicted_file):
        df_hi = pd.read_csv(predicted_file)
        df_hi["time"] = pd.to_datetime(df_hi["time"])
        df_hi["date"] = df_hi["time"].dt.date
        df_hi["hour"] = df_hi["time"].dt.hour

        hourly_for_date = df_hi[df_hi["date"] == target_date]
        if not hourly_for_date.empty:
            print(f"Found {len(hourly_for_date)} hourly predictions for {target_date}.")
        else:
            print(f"No hourly predictions found for {target_date}. Using recent averages.")
    else:
        print("No predicted_heat_index.csv found.")


    # Google Maps travel time
    gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
    try:
        directions = gmaps.directions(origin, destination, mode="driving", departure_time="now")
        if not directions:
            travel_time = timedelta(minutes=30)
            travel_time_str = "30 minutes (fallback)"
        else:
            travel_seconds = directions[0]["legs"][0]["duration"]["value"]
            travel_time = timedelta(seconds=travel_seconds)
            travel_time_str = str(travel_time).split('.')[0]
    except Exception as e:
        print("Error getting directions:", e)
        travel_time = timedelta(minutes=30)
        travel_time_str = "30 minutes (fallback due to error)"

    latest_departure = arrival_dt - travel_time
    suggested_departures = []
    intervals = [timedelta(minutes=60), timedelta(minutes=30), timedelta(minutes=0)]
    for interval in intervals:
        dep_time = latest_departure - interval
        if dep_time.date() == arrival_dt.date() and dep_time < arrival_dt:
            suggested_departures.append(dep_time)
        if len(suggested_departures) >= 3:
            break

    # Gemini setup
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel("gemini-2.0-flash-lite")

    departure_predictions = []
    gemini_recommendations = []

    for dep in suggested_departures:
        h = dep.hour
        if h in hourly_avg['hour'].values:
            base_data = hourly_avg[hourly_avg['hour'] == h].iloc[0]
            features = pd.DataFrame([{
                'temperature_2m (°C)': base_data['temperature_2m (°C)'],
                'relative_humidity_2m (%)': base_data['relative_humidity_2m (%)'],
                'rain (mm)': base_data['rain (mm)'],
                'wind_speed_10m (km/h)': base_data['wind_speed_10m (km/h)'],
                'hour': h, 'day': dep.day, 'month': dep.month,
                'is_weekend': int(dep.weekday() >= 5),
                'sin_hour': np.sin(2*np.pi*h/24),
                'cos_hour': np.cos(2*np.pi*h/24)
            }])

            # Use precomputed prediction if available, else model.predict()
            if hourly_for_date is not None and not hourly_for_date.empty:
                match = hourly_for_date[hourly_for_date["hour"] == h]
                if not match.empty:
                    hi_pred = float(match["predicted_heat_index"].iloc[0])
                else:
                    df = hourly_for_date.sort_values("hour")
                    hi_pred = np.interp(
                        dep.hour + dep.minute / 60,
                        df["hour"],
                        df["predicted_heat_index"]
                    )
            else:
                hi_pred = float(model.predict(features)[0])

            departure_predictions.append((dep, hi_pred))

            prompt = f"""You are a travel assistant. Based on the following details, give a short, focused travel recommendation:
- Departure time: {dep.strftime('%H:%M')}
- Arrival time: {arrival_dt.strftime('%H:%M')}
- Travel time: {travel_time_str}
- Heat index: {hi_pred:.2f}°C
- Origin: {origin}, Destination: {destination}, Date: {arrival_dt.strftime('%Y-%m-%d')}
Provide only relevant advice and tips specific to this departure time. Limit to 1-2 sentences."""
            try:
                response = model_gemini.generate_content(prompt)
                rec_text = getattr(response, "text", "").strip() or response.content[0].text
            except Exception as e:
                rec_text = "No AI recommendation available at the moment."
                print("Gemini error:", e)

            gemini_recommendations.append({
                "departure_time": dep.strftime('%H:%M'),
                "heat_index": f"{hi_pred:.2f}°C",
                "recommendation": rec_text
            })

    # Visualization
    hours_list = list(range(24))
    heat_list = []

    for h in hours_list:
        match = hourly_for_date[hourly_for_date["hour"] == h] if hourly_for_date is not None else pd.DataFrame()
        if not match.empty:
            heat_list.append(float(match["predicted_heat_index"].iloc[0]))
        else:
            # interpolate from model if missing
            features = pd.DataFrame([{
                'temperature_2m (°C)': hourly_avg.loc[hourly_avg['hour'] == h, 'temperature_2m (°C)'].values[0],
                'relative_humidity_2m (%)': hourly_avg.loc[hourly_avg['hour'] == h, 'relative_humidity_2m (%)'].values[0],
                'rain (mm)': hourly_avg.loc[hourly_avg['hour'] == h, 'rain (mm)'].values[0],
                'wind_speed_10m (km/h)': hourly_avg.loc[hourly_avg['hour'] == h, 'wind_speed_10m (km/h)'].values[0],
                'hour': h, 'day': target_date.day, 'month': target_date.month,
                'is_weekend': int(target_date.weekday() >= 5),
                'sin_hour': np.sin(2*np.pi*h/24),
                'cos_hour': np.cos(2*np.pi*h/24)
            }])
            hi_pred = float(model.predict(features)[0])
            heat_list.append(hi_pred)


    plt.figure(figsize=(12, 5))
    plt.plot(hours_list, heat_list, marker='o', color='#ff914d', label='Hourly Predicted Heat Index')

    for dep, hi in departure_predictions:
        dep_hour_float = dep.hour + dep.minute / 60
        plt.scatter(dep_hour_float, hi, s=90, zorder=5)
        plt.text(dep_hour_float + 0.05, hi + 0.25, f"{dep.strftime('%H:%M')}", fontsize=9, weight='bold')

    plt.xlabel("Hour of Day")
    plt.ylabel("Heat Index (°C)")
    plt.title(f"Hourly Heat Index Forecast ({arrival_dt.strftime('%Y-%m-%d')})\n{origin} ➜ {destination}")
    plt.xticks(range(0, 24))
    plt.xlim(-0.5, 23.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return jsonify({
        "graph_image": f"data:image/png;base64,{graph_base64}",
        "recommendations": gemini_recommendations,
        "travel_time": travel_time_str
    })


@app.route("/retrain", methods=["POST"])
def retrain_endpoint():
    """
    Force retrain the model synchronously.
    Accepts optional JSON {"force": true}
    """
    try:
        body = request.get_json(silent=True) or {}
        force = bool(body.get("force", True))
        initialize_model(force_retrain=force)
        return jsonify({"status": "retrained", "meta": model_meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- startup ----------
if __name__ == "__main__":
    initialize_model(force_retrain=False)
    app.run(debug=True, use_reloader=False)
