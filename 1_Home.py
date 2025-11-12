import streamlit as st
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
set_background("assets/background_1.jpg")

st.set_page_config(
    page_title="HOME | Predictive Cooling Optimizer",
    page_icon="assets/logo.png",
    layout="wide"
)

st.logo("assets/logo.png")

# page_bg_img = '''
# <style>
# body {
# background-image: url('https://wallpapers.com/images/high/modern-data-center-infrastructure-azgwkzrjo2ohl4gd.webp');
# background-size: cover;
# opacity: 0.8;
# }
# </style>
# '''

# st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("❄️ Predictive Cooling Optimizer for Data Centers")
st.markdown("### Temperature-Aware Chiller Scheduling to Cut Energy Use")

st.markdown("""
This application is a prototype frontend for a machine learning-based system to optimize data center cooling. 
It uses predictive analytics to forecast energy usage and temperature, allowing for smarter chiller scheduling to reduce energy consumption.

The core of this system is two **XGBoost regression models**:
1.  **Energy Prediction Model**: Forecasts energy consumption (kWh).
2.  **Temperature Forecasting Model**: Predicts internal temperatures one hour in advance.

Navigate the pages in the sidebar to see the system in action.
""")

st.header("🚀 Project Performance Metrics")
st.write("The models were trained on 13,615 HVAC samples and validated extensively.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Energy Prediction Model")
    st.metric(label="R² Score", value="0.9891", delta="High Accuracy")
    st.metric(label="Mean Absolute Error (MAE)", value="1.222 kWh")
    st.metric(label="Training Time", value="2.12 seconds")

with col2:
    st.subheader("Temperature Forecasting Model")
    st.metric(label="R² Score", value="0.6853")
    st.metric(label="Prediction Tolerance", value="89.24% within ±1°C")
    st.metric(label="Training Time", value="1.87 seconds")

st.header("📖 How to Use This")
st.info(
    """
    * **Live Dashboard**: See the models make predictions in real-time against the provided `sample_test_data.csv`.
    * **Manual Prediction**: Enter your own parameters to get a custom energy forecast and optimization suggestion.
    * **Model Details**: Learn more about the 46 features  and testing methodology used.
    """
)
