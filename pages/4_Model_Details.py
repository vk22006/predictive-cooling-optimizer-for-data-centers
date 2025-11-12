import streamlit as st
import pickle
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

st.set_page_config(page_title="MODEL DETAILS | Predictive Cooling Optimizer", page_icon="assets/logo.png", layout="wide")
st.logo("assets/logo.png")

st.title("⚙️ Model & Feature Details")

# --- Performance ---
st.header("1. Model Performance")
st.markdown("Performance metrics as reported in the project abstract.")

st.subheader("Energy Prediction Model")
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", "0.9891")
col2.metric("Mean Absolute Error (MAE)", "1.222 kWh")
col3.metric("Training Time", "2.12 s")

st.subheader("Temperature Forecasting Model")
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", "0.6853")
col2.metric("Prediction Tolerance", "89.24% within ±1°C")
col3.metric("Training Time", "1.87 s")

# --- Features ---
st.header("2. Feature Engineering")
st.markdown("""
The models do not use raw data. Instead, they use **46 engineered features** 
created from the raw sensor and time data. This captures complex system dynamics 
and temporal patterns.
""")

try:
    with open('models/feature_list.pkl', 'rb') as f:
        feature_list = pickle.load(f)
    
    with st.expander("Click to see all 46 features "):
        col1, col2, col3 = st.columns(3)
        # Split list into 3 columns
        split_size = (len(feature_list) + 2) // 3
        lists = [feature_list[i:i + split_size] for i in range(0, len(feature_list), split_size)]
        
        for i, sublist in enumerate(lists):
            with locals()[f'col{i+1}']:
                for item in sublist:
                    st.markdown(f"`{item}`")

except FileNotFoundError:
    st.error("`feature_list.pkl` not found. Cannot display feature list.")

# --- Testing ---
st.header("3. System Testing & Validation")
st.markdown("The system prototype underwent comprehensive testing to ensure reliability and accuracy.")

st.metric("Test Success Rate", "100%", "11/11 Tests Passed")

st.markdown("""
Five testing methodologies were used:
* **Unit Testing**: Validated the individual performance of the Energy and Temperature models.
* **Integration Testing**: Verified the data preprocessing and feature engineering pipeline.
* **Functional Testing**: Ensured accuracy and latency requirements were met.
* **White Box Testing**: Verified hyperparameter tuning and model logic.
* **Black Box Testing**: Checked boundary conditions and system consistency.
""")