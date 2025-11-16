# Aldaw-Wave

**Aldaw Wave** is an **AI-assisted weather** and **mobility decision platform** that predicts **heat index conditions** and **recommends safe departure times** for commuters. 
By combining machine learning, route intelligence, and real-time weather forecasting, Aldaw Wave helps users reduce heat exposure while traveling.

---

## Features
-   🔹**Heat Index Forecasting** --- Predicts hourly heat index using
    weather patterns and environmental inputs.\
-   🔹**Smart Travel Recommendations** --- Generates top 5 departure
    options based on heat index and travel time.\
-   🔹**AI-Generated Commuting Advice** --- Uses Google Gemini to generate
    concise, plain-text travel tips.\
-   🔹**Route Estimation** --- Integrates Google Directions API for travel
    duration.\
-   🔹**Dynamic Visualization** --- Creates heat index graphs for a full
    24-hour timeline.\
-   🔹**Fallback Historical Logic** --- Pulls 7-day historical data when
    future forecasts are unavailable.\
-   🔹**Frontend Auto-Works in Chrome** --- Fully functional by opening
    `index.html` directly (no Live Server required).

---

## How It Works

1. **Travel Time & Geolocation**
   - The system sends the user’s origin and destination to **Google Maps API.**
   - It retrieves estimated travel duration and geographic coordinates.

2. **Weather Data Retrieval**
   - Aldaw Wave requests 24-hour hourly environmental data (temperature & humidity) from Open-Meteo for the selected date.
   - If future data is not available, the system automatically gathers up to 7 previous days of historical heat-index data.

3. **Heat Index Calculation**
   - Using NOAA’s heat index formula, Aldaw Wave computes hourly heat index from temperature and humidity.
   - Missing or incomplete values are resolved through linear interpolation.

4. **Departure Window Analysis**
   - Based on the user’s target arrival time, the system generates a 10-hour range of possible departure times.
   - For each possible departure time, it estimates:
    - Arrival time
    - Expected heat index at arrival
   - It then selects the top 5 safest (lowest heat-index) options.

5. **AI-Enhanced Recommendation**
   - A customized prompt is sent to Google Gemini.
   - Gemini produces a 2–3 sentence plain-text travel tip highlighting:
    - The best on-time option (Option 1)
    - The lowest heat-index option across all 5 choices
    - Why each is recommended

6. **Visualization**
   - The system generates a 24-hour heat index graph using Matplotlib.
   - The graph and full results are displayed in the web interface.

---

## 🛠️ **Technologies Used**

-   **Python & Flask** --- Backend server and API logic\
-   **Google Maps Platform** --- Geocoding and route duration\
-   **Google Gemini API** --- AI-powered travel tip generation\
-   **Open-Meteo** --- Real-time and historical weather data\
-   **Pandas / NumPy** --- Data handling and time-series processing\
-   **Matplotlib** --- Graph creation\
-   **HTML / CSS / JavaScript** --- Web client interface

---

## **File Structure**

    AldawWave/
    ├── index.html                        # Main user interface for entering trip details and viewing results
    ├── about.html                        # About page showing project purpose and team information
    ├── static/
    │   ├── css/
    │   │   └── style.css                 # Stylesheet defining layout, spacing, colors, and UI responsiveness
    │   ├── js/
    │   │   └── script.js                 # Frontend script sending user input to backend and updating the UI
    │   ├── assets/
    │   │   └── (images...)               # Contains logos, icons, and UI-related graphic files
    ├── model/
    │   ├── dataset/
    │   │   ├── philippines_cities.csv    # Dataset of PH cities containing latitude and longitude data
    │   │   ├── heat_index.csv            # Collected historical and processed heat index values
    │   │   └── heat_index_prediction.csv # Storage for merged or fallback-generated heat index values
    │   ├── aldaw_wave.py                 # Backend engine: forecasting, routes, logic, and AI prompt processing
    │   ├── requirements.txt              # Complete list of Python dependencies for backend execution
    │   └── .env                          # Environment variables including Google and Gemini API keys (not included in github)
    └── README.md                         # Full project documentation

---

## Environment Setup (`.env` File)

The `.env` file **must be placed inside the `model/` folder**.  
This file contains sensitive environment variables such as **API keys** (for Google Maps & Gemini), which are essential for the backend services to function properly.

If you need access to the .env file for testing purposes, it may be requested directly from the developers.

### Why `.env` is NOT included in the public repository
- The `.env` file contains **private API keys** and **confidential configurations**.
- Publishing it publicly can lead to **unauthorized access**, **API abuse**, or **quota exhaustion**.
- Following best practices, the `.env` file is **listed in `.gitignore`** to prevent accidental upload to GitHub.

---


## Dependencies

All required libraries for Aldaw-Wave are listed in the `requirements.txt` file.

#### `requirements.txt`
```
Flask
flask-cors
numpy
pandas
requests
requests-cache
googlemaps
google-generativeai
python-dotenv
matplotlib

```

#### To install all dependencies:
```bash
pip install -r requirements.txt
```

---

## How to Run

### 1️. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python aldaw_wave.py
```

### 3. Open in index.html in Browser
  
Right-click → Open With → choose your browser (e.g., Chrome, Edge, Firefox)
✔ **OPEN `index.html` DIRECTLY IN CHROME**\
✘ Do **NOT** use Live Server\
✘ Do **NOT** use VSCode Preview\
✘ Do **NOT** host through extensions

---

## Example Workflow

1. User opens index.html directly in Chrome.
2. Enters origin, destination, date, and target arrival time.
3. Backend geocodes the destination using Google Maps API and retrieves travel duration.
4. System requests hourly weather data for the selected date from Open-Meteo.
5. If data for that date is unavailable:
  - Aldaw Wave automatically gathers up to 7 historical days of weather data for fallback.
  - Computes hourly average heat indexes.
6. The backend calculates:
  - Heat index per hour
  - Best 5 departure options with the lowest predicted heat index
  - AI travel advice using Google Gemini
7. A heat-index graph is generated and returned to the frontend.
8. The web app displays:
  - Graph
  - Top 5 departure recommendations
  - Estimated travel time
  - AI-generated travel tip
  - Heat index at arrival time

---

## Team Members

**Project Name:** Aldaw Wave  
**Institution/Organization:** School of Computing - HAU

**Purpose:**  
Integration of machine learning and environmental analysis to predict heat index trends for public use.

---

### Team Roles

- **Project Leader:** John Reshley P. Gonzales
  Oversees the entire development process, ensures coordination among members, and manages deadlines.

- **Assistant Leader:** Grant Mihkael D. Quilantang
  Supports the leader in project planning and task delegation, ensuring smooth team communication.

- **UI/UX Designer:** Keith Ryan N. Almanzor
  Designs the interface and user experience, focusing on accessibility, aesthetics, and usability.

- **Backend Developer:** Mark Harold T. Valderrama
  Handles server logic, machine learning integration, and database operations for reliable data management.

- **Frontend Developer:**  Randel Angel L. Yumul
  Implements the visual design into functional web components, ensuring responsiveness and interactivity.

---

## Future Enhancements

- Mobile-responsive frontend
- Improved model accuracy with new training datasets
- Cloud database support (e.g., PostgreSQL or Firebase)

---

> *Aldaw-Wave: Where Data Meets the Weather — Predicting Tomorrow’s Heat Today.*
