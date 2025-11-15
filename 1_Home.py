'''**********************************************************************************
    SPDX-License-Identifier: MIT
    Copyright (c) 2025 Kishore. V

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
**********************************************************************************'''

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

