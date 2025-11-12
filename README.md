# Predictive Cooling Optimizer for Data Centers: Temperature-Aware Chiller Scheduling to Cut Energy Use
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Languages](https://img.shields.io/github/languages/top/vk22006/predictive-cooling-optimizer-for-data-centers)](https://github.com/vk22006/predictive-cooling-optimizer-for-data-centers)

The project addresses energy inefficiency in data center cooling systems by developing a temperature-aware predictive model that optimizes chiller scheduling to reduce energy consumption while maintaining thermal safety. Traditional reactive cooling systems respond to temperature changes after they occur, leading to energy waste and suboptimal chiller operation.

## Project Methodology
  The methodology began with comprehensive data preprocessing of 13,615 HVAC samples, including outlier detection using IQR, normalization via MinMaxScaler, and chronological 80-20 train-test splitting to preserve temporal integrity. Feature engineering created 46 enhanced features encompassing lag features (16), rolling averages (12), cyclical temporal encodings (6), and interaction features (4), capturing complex system dynamics.

  Two XGBoost regression models formed the core prediction engine: an Energy Prediction Model achieving R² = 0.9891 with MAE of 1.222 kWh, and a Temperature Forecasting Model achieving R² = 0.6853 with 89.24% predictions within ±1°C tolerance. Both models demonstrated efficient training times (2.12 seconds for energy, 1.87 seconds for temperature) suitable for real-time deployment.

  A PredictiveCoolingOptimizer class integrated both models, enabling system-wide optimization through constraint-based temperature management and energy minimization strategies.

## Tools used
1. Anaconda Jupyter (for model training and testing)
2. Streamlit library (for frontend implementation)
3. Joblib (for handling of `.pkl` model files)
4. Numpy
5. Pandas
6. Sklearn

## Algorithms Used
1. Prediction Algorithms
- XGBoost (Extreme Gradient Boosting)
- Random Forest Regressor

2. Supporting Algorithms
- Min-Max Normalization
- Rolling Average (for feature engineering)


