# Aldaw-Wave

**Aldaw-Wave** is a machine learning–powered weather application that predicts the **heat index** for a specific location and date.  
The system combines **real-time weather information**, **travel route data**, and **trained regression models** to provide users with accurate, date-specific heat index forecasts.  

---

## Features

- 🔹 **Machine Learning Model**
  - Predicts hourly and daily **heat index** values based on temperature, humidity, and other environmental variables.
  - Uses a combined training and prediction pipeline for real-time and stored data.

- 🔹 **Date-Specific Prediction**
  - Stores all predictions in a CSV file (`predicted_heat_index.csv`).
  - Automatically retrieves the heat index for a specific date when entered by the user.
  - Falls back to current model predictions if no saved data is found.

- 🔹 **CSV-Based Data Handling**
  - Reads and processes data from `angelesdatabase.csv`.
  - Maintains all historical and predicted data for tracking and evaluation.

- 🔹 **Visualization**
  - Plots graphs comparing **predicted** and **actual** heat index data for analysis and validation.

- 🔹 **Web Integration**
  - Designed to integrate with a frontend web interface for user input (date, time, and location).
  - Displays temperature and heat index predictions dynamically.

---

## System Overview

```
Weather Data (angelesdatabase.csv)
        │
        ▼
Data Preprocessing → Model Training → Prediction Generation
        │
        ▼
     Save to predicted_heat_index.csv
        │
        ▼
User selects a date → Retrieves matching predictions or runs new model
```

---

## How It Works

1. **Model Training**
   - The model is trained using historical weather data from `angelesdatabase.csv`.
   - It learns patterns from temperature, humidity, and other environmental attributes.

2. **Prediction Generation**
   - When the model runs, it generates heat index predictions and saves them to `predicted_heat_index.csv`.

3. **Date Matching**
   - If the user inputs a specific date:
     - The system searches for predictions that match the date.
     - If found, the corresponding hourly heat index values are displayed.
     - If not found, the system runs the model again for that date.

4. **Visualization**
   - The results are plotted and displayed in a user-friendly chart.

---

## 🧩 File Structure

```
AldawWave/
├── index.html                          # Main user interface for displaying predictions and visuals
├── about.html                          # Page showcasing team members and their roles
├── static/
│ ├── css/
│ │ └── style.css                       # Handles layout, colors, fonts, and overall website styling
│ ├── js/
│ │ └── script.js                       # Connects frontend UI with backend services and handles interactivity
│ ├── assets/
│ │ └── (image, and logo files)          # Contains images, logos, and other media used in the website
├── model/
│ ├── dataset/
│ │ ├── angelesdataset.csv              # Main dataset containing collected weather data
│ │ └── predicted_heat_index.csv        # Stores generated or saved heat index prediction results
│ ├── joblib/
│ │ ├── .gitkeep                        # Keeps the directory tracked in GitHub
│ │ ├── aldaw_wave_model.joblib         # Trained machine learning model for heat index prediction
│ │ └── aldaw_wave_model_meta.joblib    # Metadata file storing model details (e.g., last training date)
│ ├── .env                              # Environment file (contains API keys and sensitive credentials)
│ ├── aldaw_wave_service.py             # Core script handling ML logic, data preprocessing, and predictions
│ └── requirements.txt                  # Lists all Python dependencies required to run the project
└── README.md                           # Project documentation (you are here)          
```

---

## Environment Setup (`.env` File)

The `.env` file **must be placed inside the `model/` folder**.  
This file contains sensitive environment variables such as **API keys** (for Google Maps & Gemini), which are essential for the backend services to function properly.

If you need access to the .env file for testing purposes, it may be requested directly from the developers.

### ❗ Why `.env` is NOT included in the public repository
- The `.env` file contains **private API keys** and **confidential configurations**.
- Publishing it publicly can lead to **unauthorized access**, **API abuse**, or **quota exhaustion**.
- Following best practices, the `.env` file is **listed in `.gitignore`** to prevent accidental upload to GitHub.

---

## Why `joblib` Files Are NOT Included in the Public Repository

The `joblib/` folder contains the **trained machine learning models** (`.joblib` files) used by Aldaw-Wave.  
These files are **not included** in the public repository for the following reasons:

1. **Large File Size**  
   - Model files can be several megabytes in size, which can unnecessarily increase the repository’s storage and cloning time.

2. **Regeneration by the User**  
   - These files are **automatically generated** when the user runs the model training script (`aldaw_wave_service.py`).  
   - This ensures that users always have an **up-to-date and environment-specific** version of the model.

3. **Version and Environment Differences**  
   - Joblib files depend on the specific versions of libraries (e.g., scikit-learn, NumPy) installed in the environment.  
   - Sharing them publicly can cause compatibility errors on other systems.

4. **Best Practice for Reproducibility**  
   - Instead of distributing pre-trained binary files, the repository includes the **training logic and dataset**, allowing users to reproduce and validate results on their own system.

To keep the folder visible in GitHub, a **`.gitkeep`** file is included in the `joblib/` directory.

---

## Dependencies

All required libraries for Aldaw-Wave are listed in the `requirements.txt` file.

#### `requirements.txt`
```
numpy
pandas
joblib
requests-cache
retry-requests
openmeteo-requests
python-dotenv
scikit-learn
Flask
flask-cors
matplotlib
googlemaps
google-generativeai
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

### 2️. Prepare the Data
Place your `angelesdatabase.csv` file in the root folder.

### 3️. Run the Application
```bash
python aldaw_wave_service.py
```

### 4️. Open in index.html in Browser
  
Right-click → Open With → choose your browser (e.g., Chrome, Edge, Firefox)

---

## Example Workflow

1. User opens the web app.
2. Selects a date (e.g., **2024-10-05**).
3. The app searches for matching predictions in `predicted_heat_index.csv`.
4. If found → Displays hourly heat index graph for that date.  
   If not → Generates new predictions and saves them.

---

## Example Output (Console)

```
No saved predictions found for 2024-10-05, using current model.
Predicted Heat Index saved to predicted_heat_index.csv
```

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
- Real-time API integration with live weather data
- Improved model accuracy with new training datasets
- Cloud database support (e.g., PostgreSQL or Firebase)

---

> *Aldaw-Wave: Where Data Meets the Weather — Predicting Tomorrow’s Heat Today.*
