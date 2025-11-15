# SPDX-License-Identifier: MIT

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from pathlib import Path
import base64

def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
    }}
    [data-testid="stToolbar"], [data-testid="stSidebar"] {{
        opacity: 0.8;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# call it before creating UI
set_background("assets/background_2.jpg")

st.set_page_config(page_title="MANUAL PREDICTION | Predictive Cooling Optimizer", page_icon="assets/logo.png", layout="wide")
st.logo("assets/logo.png")

# Paths
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
SAMPLE_CSV = DATA_DIR / "sample_test_data.csv"
ENERGY_MODEL_PKL = MODELS_DIR / "energy_model.pkl"
TEMP_MODEL_PKL = MODELS_DIR / "temp_model.pkl"
FEATURE_LIST_PKL = MODELS_DIR / "feature_list.pkl"
ENERGY_FEATURE_PKL = MODELS_DIR / "energy_feature_list.pkl"
TEMP_FEATURE_PKL = MODELS_DIR / "temp_feature_list.pkl"

# ---------------------------
# Helpers: load models & lists
# ---------------------------
@st.cache_resource
def load_models_and_feature_lists():
    """Return (energy_model, temp_model, energy_feature_list, temp_feature_list, fallback_list)"""
    # load models
    try:
        with open(ENERGY_MODEL_PKL, "rb") as f:
            energy_model = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load energy model: {e}")
        return None, None, None, None, None

    try:
        with open(TEMP_MODEL_PKL, "rb") as f:
            temp_model = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load temp model: {e}")
        return energy_model, None, None, None, None

    # prefer explicit per-model lists if present
    energy_feature_list = None
    temp_feature_list = None
    fallback_list = None

    if ENERGY_FEATURE_PKL.exists():
        try:
            with open(ENERGY_FEATURE_PKL, "rb") as f:
                energy_feature_list = pickle.load(f)
        except Exception:
            energy_feature_list = None

    if TEMP_FEATURE_PKL.exists():
        try:
            with open(TEMP_FEATURE_PKL, "rb") as f:
                temp_feature_list = pickle.load(f)
        except Exception:
            temp_feature_list = None

    if FEATURE_LIST_PKL.exists():
        try:
            with open(FEATURE_LIST_PKL, "rb") as f:
                fallback_list = pickle.load(f)
        except Exception:
            fallback_list = None

    # if models expose feature names, use them
    energy_model_feats = getattr(energy_model, "feature_names_in_", None)
    temp_model_feats = getattr(temp_model, "feature_names_in_", None)
    if energy_model_feats is not None:
        energy_model_feats = list(energy_model_feats)
    if temp_model_feats is not None:
        temp_model_feats = list(temp_model_feats)

    if energy_feature_list is None:
        if energy_model_feats is not None:
            energy_feature_list = energy_model_feats
        elif fallback_list is not None:
            energy_feature_list = list(fallback_list)

    if temp_feature_list is None:
        if temp_model_feats is not None:
            temp_feature_list = temp_model_feats
        elif fallback_list is not None:
            temp_feature_list = list(fallback_list)

    return energy_model, temp_model, energy_feature_list, temp_feature_list, fallback_list

@st.cache_data
def load_default_history():
    """Load sample_test_data.csv as default history, create proxies used by pipeline."""
    try:
        df = pd.read_csv(SAMPLE_CSV)
        # proxy chiller energy if missing
        if "Energy_Lag_1" in df.columns and "Chiller Energy Consumption (kWh)" not in df.columns:
            df["Chiller Energy Consumption (kWh)"] = df["Energy_Lag_1"]
        # ensure Cooling Water Temperature (C) exists for lag calculations
        if "CoolingWaterTemp_Lag_1" in df.columns and "Cooling Water Temperature (C)" not in df.columns:
            df = df.rename(columns={"CoolingWaterTemp_Lag_1": "Cooling Water Temperature (C)"})
        if "Cooling Water Temperature (C)" not in df.columns:
            df["Cooling Water Temperature (C)"] = 32.0
        # Provide an 'Energy' column used by our lag/rolling calcs (alias Energy_Lag_1 if present)
        if "Energy_Lag_1" in df.columns and "Energy" not in df.columns:
            df["Energy"] = df["Energy_Lag_1"]
        return df
    except FileNotFoundError:
        st.error("`sample_test_data.csv` not found. Cannot provide default history.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading history: {e}")
        return pd.DataFrame()

# ---------------------------
# Feature engineering pipeline
# ---------------------------
def create_feature_vector(inputs, history_df, feature_list):
    """
    Create the 46-feature vector. This emulates your training pipeline.
    inputs: dict with UI inputs (timestamp, building_load, etc.)
    history_df: DataFrame of past rows (must have enough rows for rolling)
    feature_list: ordered list of features the model expects
    """
    # Build raw row used for appending to history
    raw_row = {
        'Chilled Water Rate (L/sec)': inputs.get('chilled_water_rate', 0.0),
        'Chiller Energy Consumption (kWh)': inputs.get('chiller_energy', 0.0),
        'Building Load (RT)': inputs.get('building_load', 0.0),
        'Outside Temperature (F)': inputs.get('outside_temp', 0.0),
        'Dew Point (F)': inputs.get('dew_point', 0.0),
        'Humidity (%)': inputs.get('humidity', 0.0),
        'Wind Speed (mph)': inputs.get('wind_speed', 0.0),
        'Pressure (in)': inputs.get('pressure', 0.0),
        'Energy': inputs.get('chiller_energy', 0.0),  # used for lags
        'Timestamp': inputs.get('timestamp'),
        'Cooling Water Temperature (C)': inputs.get('cooling_water_temp_proxy', 32.0),
    }

    hist = history_df.copy()
    # Ensure columns used below exist
    for col in ['Energy', 'Building Load (RT)', 'Outside Temperature (F)', 'Cooling Water Temperature (C)']:
        if col not in hist.columns:
            hist[col] = 0.0

    # Append current to history to calculate lags/rolling
    hist_with_current = pd.concat([hist, pd.DataFrame([raw_row])], ignore_index=True)

    # Need at least 13 rows to compute rolling-12; validate upstream but be safe
    if len(hist_with_current) < 13:
        raise ValueError("Not enough history to compute rolling features (need >=13 rows).")

    last_idx = len(hist_with_current) - 1

    def safe_val(idx, col):
        try:
            return float(hist_with_current.iloc[idx][col])
        except Exception:
            return 0.0

    features = {}
    # Lags
    features['Energy_Lag_1'] = safe_val(last_idx - 1, 'Energy')
    features['BuildingLoad_Lag_1'] = safe_val(last_idx - 1, 'Building Load (RT)')
    features['OutsideTemp_Lag_1'] = safe_val(last_idx - 1, 'Outside Temperature (F)')
    features['CoolingWaterTemp_Lag_1'] = safe_val(last_idx - 1, 'Cooling Water Temperature (C)')

    features['Energy_Lag_2'] = safe_val(last_idx - 2, 'Energy')
    features['BuildingLoad_Lag_2'] = safe_val(last_idx - 2, 'Building Load (RT)')
    features['OutsideTemp_Lag_2'] = safe_val(last_idx - 2, 'Outside Temperature (F)')
    features['CoolingWaterTemp_Lag_2'] = safe_val(last_idx - 2, 'Cooling Water Temperature (C)')

    features['Energy_Lag_3'] = safe_val(last_idx - 3, 'Energy')
    features['BuildingLoad_Lag_3'] = safe_val(last_idx - 3, 'Building Load (RT)')
    features['OutsideTemp_Lag_3'] = safe_val(last_idx - 3, 'Outside Temperature (F)')
    features['CoolingWaterTemp_Lag_3'] = safe_val(last_idx - 3, 'Cooling Water Temperature (C)')

    features['Energy_Lag_6'] = safe_val(last_idx - 6, 'Energy')
    features['BuildingLoad_Lag_6'] = safe_val(last_idx - 6, 'Building Load (RT)')
    features['OutsideTemp_Lag_6'] = safe_val(last_idx - 6, 'Outside Temperature (F)')
    features['CoolingWaterTemp_Lag_6'] = safe_val(last_idx - 6, 'Cooling Water Temperature (C)')

    # Rolling stats
    roll3 = hist_with_current.iloc[-4:-1]
    roll6 = hist_with_current.iloc[-7:-1]
    roll12 = hist_with_current.iloc[-13:-1]

    features['Energy_RollingAvg_3'] = float(roll3['Energy'].mean()) if not roll3['Energy'].isnull().all() else 0.0
    features['BuildingLoad_RollingAvg_3'] = float(roll3['Building Load (RT)'].mean()) if not roll3['Building Load (RT)'].isnull().all() else 0.0
    features['OutsideTemp_RollingAvg_3'] = float(roll3['Outside Temperature (F)'].mean()) if not roll3['Outside Temperature (F)'].isnull().all() else 0.0
    features['Energy_RollingStd_3'] = float(roll3['Energy'].std()) if len(roll3) > 1 else 0.0

    features['Energy_RollingAvg_6'] = float(roll6['Energy'].mean()) if not roll6['Energy'].isnull().all() else 0.0
    features['BuildingLoad_RollingAvg_6'] = float(roll6['Building Load (RT)'].mean()) if not roll6['Building Load (RT)'].isnull().all() else 0.0
    features['OutsideTemp_RollingAvg_6'] = float(roll6['Outside Temperature (F)'].mean()) if not roll6['Outside Temperature (F)'].isnull().all() else 0.0
    features['Energy_RollingStd_6'] = float(roll6['Energy'].std()) if len(roll6) > 1 else 0.0

    features['Energy_RollingAvg_12'] = float(roll12['Energy'].mean()) if not roll12['Energy'].isnull().all() else 0.0
    features['BuildingLoad_RollingAvg_12'] = float(roll12['Building Load (RT)'].mean()) if not roll12['Building Load (RT)'].isnull().all() else 0.0
    features['OutsideTemp_RollingAvg_12'] = float(roll12['Outside Temperature (F)'].mean()) if not roll12['Outside Temperature (F)'].isnull().all() else 0.0
    features['Energy_RollingStd_12'] = float(roll12['Energy'].std()) if len(roll12) > 1 else 0.0

    # cyclical time encodings
    ts = inputs.get('timestamp')
    if not hasattr(ts, "weekday"):
        try:
            ts = pd.to_datetime(ts)
        except Exception:
            ts = datetime.datetime.now()
    if hasattr(ts, "weekday"):
        dow = ts.weekday()
    elif hasattr(ts, "dayofweek"):
        dow = int(ts.dayofweek)
    else:
        dow = 0
    hour = ts.hour if hasattr(ts, "hour") else 12
    month = ts.month if hasattr(ts, "month") else 1

    features['Hour_Sin'] = np.sin(2 * np.pi * hour / 24)
    features['Hour_Cos'] = np.cos(2 * np.pi * hour / 24)
    features['DayOfWeek_Sin'] = np.sin(2 * np.pi * dow / 7)
    features['DayOfWeek_Cos'] = np.cos(2 * np.pi * dow / 7)
    features['Month_Sin'] = np.sin(2 * np.pi * month / 12)
    features['Month_Cos'] = np.cos(2 * np.pi * month / 12)

    # interactions
    bload = raw_row['Building Load (RT)']
    outtemp = raw_row['Outside Temperature (F)']
    features['Load_Temp_Interaction'] = bload * outtemp
    features['ChilledWater_Load_Interaction'] = raw_row['Chilled Water Rate (L/sec)'] * bload
    features['Temp_Humidity_Interaction'] = outtemp * raw_row['Humidity (%)']
    features['Hour_Load_Interaction'] = hour * bload

    # include raw fields if required by feature_list
    for k, v in raw_row.items():
        if k in feature_list and k not in features:
            features[k] = v

    features_df = pd.DataFrame([features]).fillna(0.0)
    final = pd.DataFrame(columns=feature_list)
    final = pd.concat([final, features_df], ignore_index=True).fillna(0.0)
    return final[feature_list]

# ---------------------------
# Load everything
# ---------------------------
energy_model, temp_model, energy_feature_list, temp_feature_list, fallback_list = load_models_and_feature_lists()
history_df = load_default_history()

if energy_model is None or temp_model is None or history_df.empty:
    st.stop()

# ---------------------------
# UI
# ---------------------------
st.title("🔮 Manual Prediction & Optimization")
st.markdown("Enter custom inputs to forecast energy and get optimization suggestions.")

st.sidebar.header("Inputs & History")
col1, col2 = st.sidebar.columns(2)
d = col1.date_input("Date", datetime.date.today())
t = col2.time_input("Time", datetime.datetime.now().time())
ts = datetime.datetime.combine(d, t)

st.sidebar.subheader("Core Inputs")
building_load = st.sidebar.number_input("Building Load (RT)", value=506.0)
outside_temp = st.sidebar.number_input("Outside Temperature (F)", value=82.0)
humidity = st.sidebar.number_input("Humidity (%)", value=79.0)

with st.sidebar.expander("Advanced Inputs"):
    chilled_water_rate = st.number_input("Chilled Water Rate (L/sec)", value=94.0)
    chiller_energy = st.number_input("Chiller Energy Consumption (kWh)", value=132.0)
    cooling_water_temp_proxy = st.number_input("Cooling Water Temperature (C) (proxy for lags)", value=32.4)
    dew_point = st.number_input("Dew Point (F)", value=75.0)
    wind_speed = st.number_input("Wind Speed (mph)", value=12.0)
    pressure = st.number_input("Pressure (in)", value=29.8)

st.sidebar.subheader("Historical Data (optional)")
uploaded_file = st.sidebar.file_uploader("Upload Historical Data (CSV)", type="csv")
if uploaded_file is not None:
    try:
        user_hist = pd.read_csv(uploaded_file)
        if 'Energy_Lag_1' in user_hist.columns and 'Energy' not in user_hist.columns:
            user_hist['Energy'] = user_hist['Energy_Lag_1']
        if 'CoolingWaterTemp_Lag_1' in user_hist.columns and 'Cooling Water Temperature (C)' not in user_hist.columns:
            user_hist = user_hist.rename(columns={'CoolingWaterTemp_Lag_1': 'Cooling Water Temperature (C)'})
        history_df = user_hist
        st.sidebar.success("Custom history loaded.")
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded file: {e}")
        history_df = history_df
else:
    st.sidebar.info("Using default sample history for lag/rolling proxies.")

# ---------------------------
# Run pipeline on button press
# ---------------------------
if st.button("Run Prediction & Optimization"):
    # check history length
    if len(history_df) < 13:
        st.error(f"Need at least 13 rows of history to compute rolling features; current history length = {len(history_df)}.")
    else:
        inputs = {
            'timestamp': ts,
            'chilled_water_rate': chilled_water_rate,
            'chiller_energy': chiller_energy,
            'building_load': building_load,
            'outside_temp': outside_temp,
            'dew_point': dew_point,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'pressure': pressure,
            'cooling_water_temp_proxy': cooling_water_temp_proxy
        }

        # --- Build feature vector for energy model and predict energy ---
        try:
            # choose which feature_list to use to build vector for energy model
            energy_feats_for_vector = energy_feature_list if energy_feature_list is not None else (fallback_list if fallback_list is not None else None)
            if energy_feats_for_vector is None:
                st.error("No energy feature list (can't build input).")
                st.stop()
            fv_for_energy = create_feature_vector(inputs, history_df, energy_feats_for_vector)
        except Exception as e:
            st.error(f"Error during feature engineering for energy model: {e}")
            st.exception(e)
            st.stop()

        try:
            pred_energy = float(energy_model.predict(fv_for_energy)[0])
        except Exception as e:
            st.error(f"Energy model predict failed: {e}")
            st.sidebar.write("Energy model expected features:", getattr(energy_model, "feature_names_in_", None))
            st.sidebar.write("Prepared energy features (first 20):", list(fv_for_energy.columns)[:20])
            st.sidebar.write("Prepared energy sample:", fv_for_energy.iloc[0].to_dict())
            pred_energy = 0.0

        # --- Build a separate feature vector for temp model (inject predicted energy) ---
        try:
            temp_cols = temp_feature_list if temp_feature_list is not None else (fallback_list if fallback_list is not None else None)
            if temp_cols is None:
                raise RuntimeError("No temp feature list available to compose temp input.")

            # copy overlapping engineered features from energy vector if present
            base_row = {}
            for c in fv_for_energy.columns:
                if c in temp_cols:
                    base_row[c] = fv_for_energy.iloc[0][c]

            # Ensure required temperature-specific fields are present
            # 1) inject predicted energy into correct column if expected
            if 'Chiller Energy Consumption (kWh)' in temp_cols:
                base_row['Chiller Energy Consumption (kWh)'] = pred_energy

            # 2) ensure Cooling Water Temperature (C) exists (from proxy or history)
            if 'Cooling Water Temperature (C)' in temp_cols and 'Cooling Water Temperature (C)' not in base_row:
                if inputs.get('cooling_water_temp_proxy') is not None:
                    base_row['Cooling Water Temperature (C)'] = inputs.get('cooling_water_temp_proxy')
                elif 'Cooling Water Temperature (C)' in history_df.columns:
                    base_row['Cooling Water Temperature (C)'] = history_df.iloc[-1].get('Cooling Water Temperature (C)', 0.0)
                elif 'CoolingWaterTemp_Lag_1' in history_df.columns:
                    base_row['Cooling Water Temperature (C)'] = history_df.iloc[-1].get('CoolingWaterTemp_Lag_1', 0.0)
                else:
                    base_row['Cooling Water Temperature (C)'] = 0.0

            # Fill remaining temp columns from history or safe defaults
            for col in temp_cols:
                if col not in base_row:
                    if col in history_df.columns:
                        base_row[col] = history_df.iloc[-1].get(col, 0.0)
                    else:
                        # simple sensible defaults for common patterns
                        if col == 'Hour_Sin':
                            hour_val = ts.hour if hasattr(ts, "hour") else 12
                            base_row[col] = np.sin(2 * np.pi * hour_val / 24)
                        elif col == 'Hour_Cos':
                            hour_val = ts.hour if hasattr(ts, "hour") else 12
                            base_row[col] = np.cos(2 * np.pi * hour_val / 24)
                        elif col == 'DayOfWeek_Sin' or col == 'DayOfWeek_Cos':
                            dow = ts.weekday() if hasattr(ts, "weekday") else 0
                            base_row[col] = np.sin(2 * np.pi * dow / 7) if 'Sin' in col else np.cos(2 * np.pi * dow / 7)
                        elif col == 'Month_Sin':
                            m = ts.month if hasattr(ts, "month") else 1
                            base_row[col] = np.sin(2 * np.pi * m / 12)
                        elif col == 'Month_Cos':
                            m = ts.month if hasattr(ts, "month") else 1
                            base_row[col] = np.cos(2 * np.pi * m / 12)
                        else:
                            base_row[col] = 0.0

            # Build DataFrame ordered exactly as temp_cols
            fv_for_temp = pd.DataFrame([base_row], columns=temp_cols).fillna(0.0)

        except Exception as e:
            st.error(f"Error preparing temp features: {e}")
            st.exception(e)
            st.stop()

        # --- Predict temperature ---
        try:
            pred_temp = float(temp_model.predict(fv_for_temp)[0])
        except Exception as e:
            st.error(f"Temp model predict failed: {e}")
            st.sidebar.write("Temp model expected features:", getattr(temp_model, "feature_names_in_", None))
            st.sidebar.write("Prepared temp features (first 20):", list(fv_for_temp.columns)[:20])
            st.sidebar.write("Prepared temp sample:", fv_for_temp.iloc[0].to_dict())
            pred_temp = 0.0

        # ---------------------------
        # Display results + simple optimization (mock)
        # ---------------------------
        st.subheader("2. Prediction Display")
        c1, c2 = st.columns(2)
        c1.metric("⚡ Energy Consumption Forecast", f"{pred_energy:.2f} kWh")
        c2.metric("🌡️ Temperature Forecast (1hr)", f"{pred_temp:.2f} °C")

        st.subheader("3. Optimization Results")
        if pred_temp > 32.5:
            st.warning("⚠️ Predicted temp is high. Recommend stronger cooling.")
            optimized_energy = pred_energy * 1.05
            suggestion = "Increase cooling"
        else:
            st.success("✅ System stable. Suggest slight reduction.")
            optimized_energy = pred_energy * 0.96
            suggestion = "Slightly reduce chiller rate"

        energy_savings = pred_energy - optimized_energy
        st.metric("💡 Energy Savings Estimate (mock)", f"{energy_savings:.2f} kWh", delta=f"{(energy_savings/pred_energy*100) if pred_energy>0 else 0:.2f}%")
        cost_per_kwh = st.number_input("Enter cost per kWh ($)", min_value=0.0, value=0.12, format="%.4f")
        cost_savings = energy_savings * cost_per_kwh
        st.metric("💰 Cost Reduction Estimate (mock)", f"${cost_savings:.2f}")

        st.markdown(f"**Suggested action:** {suggestion}")

