# Predictive Cooling Optimizer for Data Centers: Temperature-Aware Chiller Scheduling to Cut Energy Use
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The project addresses energy inefficiency in data center cooling systems by developing a temperature-aware predictive model that optimizes chiller scheduling to reduce energy consumption while maintaining thermal safety. Traditional reactive cooling systems respond to temperature changes after they occur, leading to energy waste and suboptimal chiller operation.

![home page](img/home_page.PNG 'home page')

## Project Methodology
  The methodology began with comprehensive data preprocessing of 13,615 HVAC samples, including outlier detection using IQR, normalization via MinMaxScaler, and chronological 80-20 train-test splitting to preserve temporal integrity. Feature engineering created 46 enhanced features encompassing lag features (16), rolling averages (12), cyclical temporal encodings (6), and interaction features (4), capturing complex system dynamics.

  Two XGBoost regression models formed the core prediction engine: an Energy Prediction Model achieving R² = 0.9891 with MAE of 1.222 kWh, and a Temperature Forecasting Model achieving R² = 0.6853 with 89.24% predictions within ±1°C tolerance. Both models demonstrated efficient training times (2.12 seconds for energy, 1.87 seconds for temperature) suitable for real-time deployment.

  A PredictiveCoolingOptimizer class integrated both models, enabling system-wide optimization through constraint-based temperature management and energy minimization strategies.

## Testing
There are a total of 11 tests conducted under five categories. Here's the details:

|       Tests       |                       Target                       | Status   |
|:-----------------:|:--------------------------------------------------:|----------|
| Unit Tests        | Energy and Temperature models, Optimization Engine | ✅ Passed |
| Integration Tests | End-to-End Pipeline, System integration            | ✅ Passed |
| Functional Tests  | Accuracy, Response Time and Logic                  | ✅ Passed |
| White Box Test    | Hyperparameters, Feature Engineering               | ✅ Passed |
| Black Box Test    | Boundary Values, Output Consistency                | ✅ Passed |
|                   | Tests Passed                                       | 11/11    |
|                   | Tests Failed                                       | 0/11     |
|                   | Success Rate                                       | 100.0%   |

## Procedure for Execution
The execution of the program is as simple as it can get. Here's a step-by-step procedure on how to do so.
1. Install necessary libraries:
```bash
pip install xgboost streamlit
```
2. Navigate to the project folder in Command Prompt/Powershell:
```bash
cd <your-file-path>
```
3. Run the application using this command:
```bash
streamlit run 1_Home.py
```

## Tools used
1. Anaconda Jupyter (for model training and testing)
2. Streamlit library (for frontend implementation)
3. Joblib (for handling of `.pkl` model files)
4. Numpy
5. Pandas
6. Sklearn
7. XGBoost

## Algorithms Used
1. Prediction Algorithms
- XGBoost (Extreme Gradient Boosting)
- Random Forest Regressor

2. Supporting Algorithms
- Min-Max Normalization
- Rolling Average (for feature engineering)

The tool successfully demonstrates feasibility for software-based predictive cooling optimization, with models ready for deployment in interactive web applications using Streamlit for user-freindly interface and stakeholder presentation.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

