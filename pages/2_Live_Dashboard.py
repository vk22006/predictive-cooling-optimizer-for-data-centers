# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Kishore V

import streamlit as st
import pandas as pd
import pickle
import time
import plotly.graph_objects as go
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

st.set_page_config(page_title="LIVE DASHBOARD | Predictive Cooling Optimizer", page_icon="assets/logo.png", layout="wide")
st.logo("assets/logo.png")

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
SAMPLE_CSV = DATA_DIR / "sample_test_data.csv"
ENERGY_MODEL_PKL = MODELS_DIR / "energy_model.pkl"
TEMP_MODEL_PKL = MODELS_DIR / "temp_model.pkl"
FEATURE_LIST_PKL = MODELS_DIR / "feature_list.pkl"
ENERGY_FEATURE_PKL = MODELS_DIR / "energy_feature_list.pkl"
TEMP_FEATURE_PKL = MODELS_DIR / "temp_feature_list.pkl"

# ---------------------------
# Load helpers
# ---------------------------
@st.cache_resource
def load_models_and_feature_lists():
    """Load models and best-effort feature lists for each model."""
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

    # Prefer explicit per-model feature lists if present
    energy_feature_list = None
    temp_feature_list = None
    try:
        if ENERGY_FEATURE_PKL.exists():
            with open(ENERGY_FEATURE_PKL, "rb") as f:
                energy_feature_list = pickle.load(f)
    except Exception:
        energy_feature_list = None

    try:
        if TEMP_FEATURE_PKL.exists():
            with open(TEMP_FEATURE_PKL, "rb") as f:
                temp_feature_list = pickle.load(f)
    except Exception:
        temp_feature_list = None

    # If models expose feature names directly, use them (best)
    energy_model_feats = getattr(energy_model, "feature_names_in_", None)
    temp_model_feats = getattr(temp_model, "feature_names_in_", None)
    # convert numpy arrays to lists if necessary
    if energy_model_feats is not None:
        energy_model_feats = list(energy_model_feats)
    if temp_model_feats is not None:
        temp_model_feats = list(temp_model_feats)

    # If explicit lists not present, try fallback feature_list.pkl
    fallback_list = None
    if FEATURE_LIST_PKL.exists():
        try:
            with open(FEATURE_LIST_PKL, "rb") as f:
                fallback_list = pickle.load(f)
        except Exception:
            fallback_list = None

    # If energy_feature_list still None, pick best available source:
    if energy_feature_list is None:
        if energy_model_feats is not None:
            energy_feature_list = energy_model_feats
        elif fallback_list is not None:
            # attempt to select a subset in same order as fallback (trust fallback ordering)
            energy_feature_list = [c for c in fallback_list]
    if temp_feature_list is None:
        if temp_model_feats is not None:
            temp_feature_list = temp_model_feats
        elif fallback_list is not None:
            temp_feature_list = [c for c in fallback_list]

    return energy_model, temp_model, energy_feature_list, temp_feature_list, fallback_list

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(SAMPLE_CSV)
        # Create safe proxies for demo:
        if "Energy_Lag_1" in df.columns and "Chiller Energy Consumption (kWh)" not in df.columns:
            df["Chiller Energy Consumption (kWh)"] = df["Energy_Lag_1"]
        if "Energy_Lag_1" in df.columns:
            df["Actual_Energy_1hr_Ago"] = df["Energy_Lag_1"]
        else:
            df["Actual_Energy_1hr_Ago"] = 0.0
        if "OutsideTemp_Lag_1" in df.columns:
            df["Actual_Temp_1hr_Ago"] = df["OutsideTemp_Lag_1"]
        elif "Outside Temperature (F)" in df.columns:
            df["Actual_Temp_1hr_Ago"] = df["Outside Temperature (F)"]
        else:
            df["Actual_Temp_1hr_Ago"] = 0.0
        return df
    except FileNotFoundError:
        st.error("`sample_test_data.csv` not found in `data/` directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading sample data: {e}")
        return pd.DataFrame()

# ---------------------------
# Load
# ---------------------------
energy_model, temp_model, energy_feature_list, temp_feature_list, fallback_list = load_models_and_feature_lists()
df = load_data()

# quick sanity stop
if energy_model is None or temp_model is None or df.empty:
    st.stop()

st.sidebar.header("Diagnostics")
st.sidebar.write("Energy model features detected:", len(energy_feature_list) if energy_feature_list is not None else None)
st.sidebar.write("Temp model features detected:", len(temp_feature_list) if temp_feature_list is not None else None)
if fallback_list is not None:
    st.sidebar.write("Fallback list length:", len(fallback_list))

# If energy_feature_list expects different names from df, create missing cols with zeros
if energy_feature_list is not None:
    for col in energy_feature_list:
        if col not in df.columns:
            df[col] = 0.0

# Build X_test for the energy model only using its own feature list
X_test_energy = df[energy_feature_list].copy() if energy_feature_list is not None else df.copy()

# ---------------------------
# UI
# ---------------------------
st.title("📊 Live Data Center Dashboard")
st.markdown("Simulating live predictions using `sample_test_data.csv`.")

if "run_demo" not in st.session_state:
    st.session_state.run_demo = False

# Start/stop buttons: no experimental_rerun usage
if st.button("▶ Start Live Demo") and not st.session_state.run_demo:
    st.session_state.run_demo = True

if st.button("■ Stop Demo") and st.session_state.run_demo:
    st.session_state.run_demo = False

# KPI + charts placeholders
st.subheader("Key Predictions")
col1, col2, col3 = st.columns(3)
kpi1 = col1.empty()
kpi2 = col2.empty()
kpi3 = col3.empty()

st.subheader("Live Prediction Charts")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("#### Energy Consumption (kWh)")
    chart1_placeholder = st.empty()
with chart_col2:
    st.markdown("#### Temperature Forecast (°C)")
    chart2_placeholder = st.empty()

history_df = pd.DataFrame(columns=['Time', 'Predicted Energy', 'Actual Energy', 'Predicted Temp', 'Actual Temp'])

# Determine final temp_feature_names list we'll use
temp_feature_names = temp_feature_list if temp_feature_list is not None else (fallback_list if fallback_list is not None else [])

# ---------------------------
# Run stream loop
# ---------------------------
if st.session_state.run_demo:
    for i in range(len(X_test_energy)):
        if not st.session_state.run_demo:
            st.warning("Live demo stopped by user.")
            break

        # Energy model input row
        current_energy_features = pd.DataFrame(X_test_energy.iloc[i]).T

        # Predict energy
        try:
            pred_energy = float(energy_model.predict(current_energy_features)[0])
        except Exception as e:
            # show helpful diag for copy-paste
            st.sidebar.error("Energy model predict failed (see below).")
            st.sidebar.write("Energy model expected features (if available):", getattr(energy_model, "feature_names_in_", None))
            st.sidebar.write("Prepared energy input cols:", list(current_energy_features.columns)[:30])
            st.sidebar.write("Prepared energy input sample:", current_energy_features.iloc[0].to_dict())
            st.error(f"Energy model predict failed: {e}")
            pred_energy = 0.0

        # Build temp input matching temp_feature_names
        temp_row = {}
        for col in temp_feature_names:
            if col in df.columns:
                temp_row[col] = df.iloc[i][col]
            else:
                # mapping & proxies
                if col == 'Chiller Energy Consumption (kWh)':
                    temp_row[col] = pred_energy
                elif col == 'Cooling Water Temperature (C)':
                    if 'CoolingWaterTemp_Lag_1' in df.columns:
                        temp_row[col] = df.iloc[i]['CoolingWaterTemp_Lag_1']
                    elif 'Cooling Water Temperature (C)' in df.columns:
                        temp_row[col] = df.iloc[i]['Cooling Water Temperature (C)']
                    else:
                        temp_row[col] = 0.0
                elif col == 'Outside Temperature (F)':
                    if 'Outside Temperature (F)' in df.columns:
                        temp_row[col] = df.iloc[i]['Outside Temperature (F)']
                    elif 'OutsideTemp_Lag_1' in df.columns:
                        temp_row[col] = df.iloc[i]['OutsideTemp_Lag_1']
                    else:
                        temp_row[col] = 0.0
                else:
                    temp_row[col] = df.iloc[i].get(col, 0.0)

        # ensure ordering and build df
        try:
            X_temp = pd.DataFrame([temp_row], columns=temp_feature_names)
        except Exception:
            # fallback: create with keys sorted to avoid crash
            X_temp = pd.DataFrame([temp_row])

        # Predict temp
        try:
            pred_temp = float(temp_model.predict(X_temp)[0])
        except Exception as e:
            st.sidebar.error("Temp model predict failed - debug info below.")
            st.sidebar.write("Temp model expected features (if available):", getattr(temp_model, "feature_names_in_", None))
            st.sidebar.write("Prepared temp input cols:", list(X_temp.columns)[:50])
            st.sidebar.write("Prepared temp input sample:", X_temp.iloc[0].to_dict())
            st.error(f"Temp model predict failed: {e}")
            pred_temp = 0.0

        # Actual proxies
        actual_energy = df.iloc[i].get('Actual_Energy_1hr_Ago', df.iloc[i].get('Energy_Lag_1', 0.0))
        actual_temp = df.iloc[i].get('Actual_Temp_1hr_Ago', df.iloc[i].get('OutsideTemp_Lag_1', 0.0))

        # Update KPIs
        kpi1.metric("⚡ Energy Forecast (kWh)", f"{pred_energy:.2f}", f"{pred_energy - actual_energy:.2f} vs. actual")
        kpi2.metric("🌡️ Temp Forecast (°C)", f"{pred_temp:.2f}", f"{pred_temp - actual_temp:.2f} vs. actual")
        savings = (actual_energy - pred_energy) / actual_energy if actual_energy > 0 else 0.0
        kpi3.metric("💡 Potential Savings", f"{savings:.1%}", "Optimized vs. Actual")

        # Update history and charts
        history_df.loc[i] = [i, pred_energy, actual_energy, pred_temp, actual_temp]
        fig_energy = go.Figure()
        fig_energy.add_trace(go.Scatter(x=history_df['Time'], y=history_df['Actual Energy'], name='Actual Energy', mode='lines', line=dict(dash='dash')))
        fig_energy.add_trace(go.Scatter(x=history_df['Time'], y=history_df['Predicted Energy'], name='Predicted Energy', mode='lines'))
        fig_energy.update_layout(title="Energy: Actual vs. Predicted", xaxis_title="Time (Steps)", yaxis_title="Energy (kWh)", height=400)
        chart1_placeholder.plotly_chart(fig_energy, use_container_width=True)

        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=history_df['Time'], y=history_df['Actual Temp'], name='Actual Temp', mode='lines', line=dict(dash='dash')))
        fig_temp.add_trace(go.Scatter(x=history_df['Time'], y=history_df['Predicted Temp'], name='Predicted Temp', mode='lines'))
        fig_temp.update_layout(title="Temperature: Actual vs. Predicted", xaxis_title="Time (Steps)", yaxis_title="Temperature (°C)", height=400)
        chart2_placeholder.plotly_chart(fig_temp, use_container_width=True)

        time.sleep(1)

    st.session_state.run_demo = False
    st.success("Live demo complete.")
else:
    st.info("Press 'Start Live Demo' to see the model predictions in action.")

