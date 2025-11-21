import os
import warnings
from datetime import datetime, timedelta, date
import time
import math

import numpy as np
import pandas as pd
import requests
import requests_cache

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import Flask, send_file

import googlemaps
import google.generativeai as genai

# Ensure matplotlib uses non-interactive backend for headless servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import base64
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")

# ---------- Configuration ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../.."))  # go up to project root
PH_CITIES_CSV = os.path.join(BASE_DIR, "dataset", "philippines_cities.csv")
HEAT_INDEX_CSV = os.path.join(BASE_DIR, "dataset", "heat_index.csv")
HEAT_INDEX_PRED_CSV = os.path.join(BASE_DIR, "dataset", "heat_index_prediction.csv")
CACHE_DB = os.path.join(BASE_DIR, ".http_cache")
TIMEZONE = "Asia/Manila"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# Flask
app = Flask(__name__, static_folder="../static")
CORS(app, resources={r"/*": {"origins": "*"}})

# Cached HTTP session (reduce repeated external calls)
requests_cache.install_cache(CACHE_DB, expire_after=3600)
session = requests.Session()

# ---------- Utils ----------

def compute_heat_index_vectorized(temp_c, humidity):
    """Compute heat index (°C) from temperature (°C) and relative humidity (%) using NOAA formula."""
    temp_f = temp_c * 9/5 + 32
    hi_f = (-42.379 + 2.04901523*temp_f + 10.14333127*humidity
            - 0.22475541*temp_f*humidity - 0.00683783*(temp_f**2)
            - 0.05481717*(humidity**2) + 0.00122874*(temp_f**2)*humidity
            + 0.00085282*temp_f*(humidity**2) - 0.00000199*(temp_f**2)*(humidity**2))
    hi_c = (hi_f - 32) * 5/9
    return hi_c

def load_ph_cities(csv_path=PH_CITIES_CSV):
    """Load philippines_cities.csv; expected columns: city, latitude, longitude."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # Normalize known alternate names
    rename_map = {}
    if 'name' in df.columns and 'city' not in df.columns:
        rename_map['name'] = 'city'
    if 'lat' in df.columns and 'latitude' not in df.columns:
        rename_map['lat'] = 'latitude'
    if 'lon' in df.columns and 'longitude' not in df.columns:
        rename_map['lon'] = 'longitude'
    if rename_map:
        df = df.rename(columns=rename_map)
    # Validate
    required = {'city', 'latitude', 'longitude'}
    if not required.issubset(set(df.columns)):
        raise RuntimeError("philippines_cities.csv must contain columns: city,latitude,longitude")
    df['city_key'] = df['city'].astype(str).str.lower().str.replace('\ufffd','').str.strip()
    return df

def geocode_destination_get_city(dest):
    """Use Google Maps Geocoding to find city and lat/lon. Returns dict."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("Google API key not configured.")
    gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
    try:
        geocode_result = gmaps.geocode(dest, region='ph')
    except Exception:
        geocode_result = []
    if not geocode_result:
        try:
            geocode_result = gmaps.geocode(dest)
        except Exception as e:
            raise RuntimeError(f"Google geocode error: {e}")
        if not geocode_result:
            raise RuntimeError(f"Google Geocoding returned no result for destination: {dest}")

    res = geocode_result[0]
    loc = res.get('geometry', {}).get('location', {})
    lat = float(loc.get('lat', 0.0))
    lon = float(loc.get('lng', 0.0))

    city = None
    admin_levels = ['locality', 'postal_town', 'administrative_area_level_2', 'administrative_area_level_1']
    for comp in res.get('address_components', []):
        types = comp.get('types', [])
        if any(t in types for t in admin_levels):
            city = comp.get('long_name')
            break

    if not city:
        formatted = res.get('formatted_address', '')
        parts = [p.strip() for p in formatted.split(',') if p.strip()]
        if parts:
            city = parts[0]

    return {'city_name': city, 'google_lat': lat, 'google_lon': lon}

def find_latlon_in_phcities(city_name, ph_df):
    """Best-effort match city_name in ph_df. Returns (lat, lon, matched_city) or None."""
    if city_name is None:
        return None
    key = str(city_name).lower().strip()
    mask = ph_df['city_key'] == key
    if mask.any():
        row = ph_df[mask].iloc[0]
        return float(row['latitude']), float(row['longitude']), row['city']
    # substring or start match
    short = key.split()[0]
    mask = ph_df['city_key'].str.contains(key) | ph_df['city_key'].str.contains(short)
    if mask.any():
        row = ph_df[mask].iloc[0]
        return float(row['latitude']), float(row['longitude']), row['city']
    for start in key.split():
        mask = ph_df['city_key'].str.startswith(start)
        if mask.any():
            row = ph_df[mask].iloc[0]
            return float(row['latitude']), float(row['longitude']), row['city']
    return None

def fetch_openmeteo_hourly(lat, lon, target_date: date):
    """Fetch hourly temperature and relative humidity from Open-Meteo archive for a single date."""
    start = target_date.strftime('%Y-%m-%d')
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start,
        'end_date': start,
        'hourly': 'temperature_2m,relativehumidity_2m',
        'timezone': TIMEZONE
    }
    try:
        r = session.get(OPENMETEO_ARCHIVE, params=params, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print("Open-Meteo request error:", e)
        return pd.DataFrame()
    j = r.json()
    if 'hourly' not in j or 'time' not in j['hourly']:
        return pd.DataFrame()
    times = pd.to_datetime(j['hourly']['time']).tz_localize(None)
    temps = np.array(j['hourly'].get('temperature_2m', []), dtype=float)
    hum = np.array(j['hourly'].get('relativehumidity_2m', []), dtype=float)
    df = pd.DataFrame({'time': times, 'temperature_2m (°C)': temps, 'relative_humidity_2m (%)': hum})
    if not df.empty:
        df['hour'] = df['time'].dt.hour
    return df

def save_heat_index_df(df: pd.DataFrame, path=HEAT_INDEX_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_to_save = df.copy()
    if 'time' in df_to_save.columns:
        try:
            df_to_save['time'] = pd.to_datetime(df_to_save['time'])
            df_to_save['time'] = df_to_save['time'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        except Exception:
            pass
    if os.path.exists(path):
        try:
            existing = pd.read_csv(path)
            merged = pd.concat([existing, df_to_save], ignore_index=True).drop_duplicates(subset=['time','destination'])
            merged.to_csv(path, index=False)
        except Exception:
            df_to_save.to_csv(path, index=False)
    else:
        df_to_save.to_csv(path, index=False)
    print(f"Saved heat index data to {path} ({len(df)} rows appended)")

def save_prediction_csv(hourly_avg_df: pd.DataFrame, target_date: date, path=HEAT_INDEX_PRED_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = hourly_avg_df.copy()
    df['predicted_for_date'] = target_date.strftime('%Y-%m-%d')
    df_to_save = df[['hour','temperature_2m (°C)','relative_humidity_2m (%)','heat_index','predicted_for_date']].copy()
    df_to_save = df_to_save.rename(columns={'heat_index':'predicted_heat_index'})
    df_to_save.to_csv(path, index=False)
    print(f"Saved predicted heat index (averaged) for {target_date} to {path}")

# ---------- Endpoints ----------

@app.route("/")
def index():
    # Serve index.html from the project root
    return send_file(os.path.join(ROOT_DIR, "index.html"))

@app.route('/get_api_key', methods=['GET'])
def get_api_key():
    if not GOOGLE_API_KEY:
        return jsonify({'error': 'GOOGLE_API_KEY not configured on server'}), 500
    return jsonify({'key': GOOGLE_API_KEY})

@app.route('/recommendation', methods=['POST'])
def recommendation():
    # Validate input JSON
    try:
        body = request.get_json(force=True)
        origin = body.get('origin', '').strip()
        destination = body.get('destination', '').strip()
        date_input = body.get('date', '').strip()
        time_input = body.get('time', '').strip()
    except Exception as e:
        return jsonify({'error': f'Invalid JSON input: {e}'}), 400

    if not origin or not destination or not date_input or not time_input:
        return jsonify({'error': 'origin, destination, date and time are required.'}), 400

    try:
        arrival_dt = datetime.strptime(f"{date_input} {time_input}", "%Y-%m-%d %H:%M")
    except ValueError:
        return jsonify({'error': 'Invalid date/time format. Use YYYY-MM-DD and HH:MM.'}), 400

    # Load PH cities
    try:
        ph_df = load_ph_cities()
    except Exception as e:
        return jsonify({'error': f'City dataset error: {e}'}), 500

    # Geocode / map to PH cities
    try:
        geo = geocode_destination_get_city(destination)
    except Exception as e:
        return jsonify({'error': f'Geocoding error: {e}'}), 500

    try:
        mapped = find_latlon_in_phcities(geo.get('city_name'), ph_df)
        if mapped is None:
            lat, lon = geo['google_lat'], geo['google_lon']
            matched_city = geo.get('city_name') or 'Unknown'
        else:
            lat, lon, matched_city = mapped
    except Exception as e:
        return jsonify({'error': f'City matching error: {e}'}), 500

    target_date_obj = None
    try:
        target_date_obj = datetime.strptime(date_input, "%Y-%m-%d").date()
    except Exception:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400

    # Fetch data for target date (or fallback to historical average)
    df_day = fetch_openmeteo_hourly(lat, lon, target_date_obj)
    hourly_for_date = None

    if not df_day.empty and len(df_day) >= 24:
        df_day['heat_index'] = compute_heat_index_vectorized(df_day['temperature_2m (°C)'], df_day['relative_humidity_2m (%)'])
        df_day['destination'] = destination
        df_day['city'] = matched_city
        df_day['latitude'] = lat
        df_day['longitude'] = lon
        try:
            save_heat_index_df(df_day)
        except Exception as e:
            print("Warning: failed to save heat index csv:", e)
        hourly_for_date = df_day.copy()
    else:
        # Collect up to 7 previous complete days
        collected = []
        look_date = target_date_obj - timedelta(days=1)
        attempts = 0
        while len(collected) < 7 and attempts < 20:
            df_tmp = fetch_openmeteo_hourly(lat, lon, look_date)
            if not df_tmp.empty and len(df_tmp) >= 24:
                df_tmp['heat_index'] = compute_heat_index_vectorized(df_tmp['temperature_2m (°C)'], df_tmp['relative_humidity_2m (%)'])
                df_tmp['destination'] = destination
                df_tmp['city'] = matched_city
                df_tmp['latitude'] = lat
                df_tmp['longitude'] = lon
                collected.append(df_tmp)
            look_date -= timedelta(days=1)
            attempts += 1
            time.sleep(0.05)
        if len(collected) == 0:
            return jsonify({'error': 'No historical data available for this location/date.'}), 500
        all_collected = pd.concat(collected, ignore_index=True)
        try:
            save_heat_index_df(all_collected)
        except Exception:
            pass
        hourly_avg = all_collected.groupby('hour')[['temperature_2m (°C)','relative_humidity_2m (%)','heat_index']].mean().reset_index()
        save_prediction_csv(hourly_avg, target_date_obj)
        # Build hourly_for_date (simulate times)
        hourly_for_date = hourly_avg.copy()
        hourly_for_date['time'] = [datetime.combine(target_date_obj, datetime.min.time()) + timedelta(hours=int(h)) for h in hourly_for_date['hour']]

    # Get travel time via Google Directions (safe)
    travel_time = timedelta(minutes=30)
    travel_time_str = '30 minutes (fallback)'
    if GOOGLE_API_KEY:
        try:
            gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
            directions = gmaps.directions(origin, destination, mode='driving', departure_time='now')
            if directions and len(directions) > 0:
                travel_seconds = directions[0]['legs'][0]['duration']['value']
                travel_time = timedelta(seconds=travel_seconds)
                travel_time_str = str(travel_time).split('.')[0]
        except Exception as e:
            print("Warning: Google Directions error:", e)
            # keep fallback
    else:
        print("Warning: GOOGLE_API_KEY not set; using fallback travel time.")

    # Build departure window
    latest_departure = arrival_dt - travel_time
    window_start = latest_departure - timedelta(hours=5)
    window_end = latest_departure + timedelta(hours=5)
    num_intervals = int(((window_end - window_start).total_seconds() // 1800) + 1)
    departure_times = [window_start + timedelta(minutes=30*i) for i in range(max(0, num_intervals))]

    if 'hour' not in hourly_for_date.columns:
        if 'time' in hourly_for_date.columns:
            hourly_for_date['hour'] = hourly_for_date['time'].dt.hour
        else:
            hourly_for_date['hour'] = np.arange(len(hourly_for_date))

    hour_series = hourly_for_date['hour'].values
    hi_series = hourly_for_date['heat_index'].values
    order = np.argsort(hour_series)
    hour_series = hour_series[order]
    hi_series = hi_series[order]

    def interp_hi(dt):
        hf = dt.hour + dt.minute/60.0
        if hf >= hour_series.min() and hf <= hour_series.max():
            return float(np.interp(hf, hour_series, hi_series))
        else:
            extended_h = np.concatenate((hour_series, hour_series+24))
            extended_hi = np.concatenate((hi_series, hi_series))
            return float(np.interp(hf, extended_h, extended_hi))

    hi_list = [(dep, interp_hi(dep)) for dep in departure_times]
    # Split before and after preferred time
    before = [(dt, hi) for dt, hi in hi_list if dt <= latest_departure]
    after  = [(dt, hi) for dt, hi in hi_list if dt > latest_departure]

    # Sort each group by lowest heat index
    before_sorted = sorted(before, key=lambda x: x[1])
    after_sorted = sorted(after, key=lambda x: x[1])

    # Select Top 3 from BEFORE preferred time
    top_before = before_sorted[:3]

    # Select Top 2 from AFTER preferred time
    top_after = after_sorted[:2]

    # Final ordered list: 1–3 before, 4–5 after
    top_5 = top_before + top_after


    # Gemini AI tips (safe usage with fallback)
    gemini_recommendations = []
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model_gemini = genai.GenerativeModel("gemini-2.0-flash-lite")

        # Prepare top 5 options text
        dep_options_text = "\n".join([
            f"{i+1}. Leave at {dep.strftime('%I:%M %p')} (Predicted Heat Index: {hi:.2f}°C)"
            for i, (dep, hi) in enumerate(top_5)
        ])

        final_prompt = f"""
        You are a smart commuting advisor for the Aldaw Wave project. Your goal is to help a commuter in Angeles City avoid extreme heat.

        Here are the top 5 departure options from {origin} to {destination} with a target arrival at {arrival_dt.strftime('%H:%M')}. 
        Options 1–3 occur before the preferred time, while Options 4–5 occur after:

        {dep_options_text}

        Instructions:
        - Your response MUST mention **two recommended times**:
            1. **Option 1**, which is the best choice for arriving on time with a relatively low heat index.
            2. The **option that has the lowest heat index** among all 5 options.
        - Clearly explain **why Option 1 is good** and **why the lowest-heat-index option is good**, in 2–3 concise sentences.
        - Your answer must be actionable, simple, and helpful for the commuter.
        - Do NOT use any Markdown formatting, bold text, asterisks, or special characters such as **, *, _, or similar. 
        - Only use plain text.
        """

        try:
            response = model_gemini.generate_content(final_prompt)
            consolidated_tip = getattr(response, 'text', '') or \
                            (response.content[0].text if getattr(response, 'content', None) else
                                "Leave early to avoid peak heat; bring water and use shaded routes.")
            consolidated_tip = consolidated_tip.strip()
        except Exception:
            consolidated_tip = "Leave early to avoid peak heat; bring water and use shaded routes."
    else:
        consolidated_tip = "Leave early to avoid peak heat; bring water and use shaded routes."

    # --- Build simplified top 5 list (time + heat index only) ---
    top_5_simple = [
        {'Option': f"Option {i+1}", 
        'departure_time': dep.strftime('%I:%M %p'), 
        'heat_index': f"{hi:.2f}°C"} 
        for i, (dep, hi) in enumerate(top_5)
    ]
    # ---- Heat Index at Preferred Arrival Time ----
    arrival_heat_index = interp_hi(arrival_dt)

    # Build line graph (24 points)
    hours_list = list(range(24))
    heat_list = []
    for h in hours_list:
        if h in list(hour_series):
            idx = list(hour_series).index(h)
            heat_list.append(float(hi_series[idx]))
        else:
            t = datetime.combine(target_date_obj, datetime.min.time()) + timedelta(hours=h)
            heat_list.append(interp_hi(t))

    plt.figure(figsize=(10,4))
    plt.plot(hours_list, heat_list, marker='o',color='#ff914d', label='Hourly Heat Index')
    for dep, hi in top_5:
        dep_h = dep.hour + dep.minute/60.0
        plt.scatter(dep_h, hi, s=60, zorder=5)
        plt.text(dep_h + 0.05, hi + 0.25, dep.strftime('%H:%M'), fontsize=8)
    plt.xlabel('Hour of Day')
    plt.ylabel('Heat Index (°C)')
    plt.title(f'Hourly Heat Index ({date_input}) - {matched_city}')
    plt.xticks(range(0,24))
    plt.xlim(-0.5, 23.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return jsonify({
        'graph_image': f'data:image/png;base64,{graph_base64}',
        'travel_time': travel_time_str,
        'arrival_heat_index': f"{arrival_heat_index:.2f}°C",
        'location_city': matched_city,
        'lat': lat,
        'lon': lon,
        'consolidated_tip': consolidated_tip,
        'top_5_options': top_5_simple
    }), 200

if __name__ == '__main__':
    # Sanity checks
    if not os.path.exists(PH_CITIES_CSV):
        print('ERROR: dataset/philippines_cities.csv not found. Please add it with columns: city,latitude,longitude')
    if not GOOGLE_API_KEY:
        print('WARNING: GOOGLE_API_KEY not set. Some features will use fallbacks.')
    if not GEMINI_API_KEY:
        print('WARNING: GEMINI_API_KEY not set. AI recommendations will be basic fallbacks.')
    app.run(host="0.0.0.0", port=5000, debug=False)
