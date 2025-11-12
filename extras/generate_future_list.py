import pickle

# This list is from your ValueError message.
# It is what your saved XGBoost models expect.
correct_feature_list = [
    'Chilled Water Rate (L/sec)', 'Building Load (RT)', 'Chiller Energy Consumption (kWh)', 
    'Outside Temperature (F)', 'Dew Point (F)', 'Humidity (%)', 'Wind Speed (mph)', 
    'Pressure (in)', 'Energy_Lag_1', 'BuildingLoad_Lag_1', 'OutsideTemp_Lag_1', 
    'CoolingWaterTemp_Lag_1', 'Energy_Lag_2', 'BuildingLoad_Lag_2', 
    'OutsideTemp_Lag_2', 'CoolingWaterTemp_Lag_2', 'Energy_Lag_3', 
    'BuildingLoad_Lag_3', 'OutsideTemp_Lag_3', 'CoolingWaterTemp_Lag_3', 
    'Energy_Lag_6', 'BuildingLoad_Lag_6', 'OutsideTemp_Lag_6', 
    'CoolingWaterTemp_Lag_6', 'Energy_RollingAvg_3', 'BuildingLoad_RollingAvg_3', 
    'OutsideTemp_RollingAvg_3', 'Energy_RollingStd_3', 'Energy_RollingAvg_6', 
    'BuildingLoad_RollingAvg_6', 'OutsideTemp_RollingAvg_6', 'Energy_RollingStd_6', 
    'Energy_RollingAvg_12', 'BuildingLoad_RollingAvg_12', 
    'OutsideTemp_RollingAvg_12', 'Energy_RollingStd_12', 'Hour_Sin', 'Hour_Cos', 
    'DayOfWeek_Sin', 'DayOfWeek_Cos', 'Month_Sin', 'Month_Cos', 
    'Load_Temp_Interaction', 'ChilledWater_Load_Interaction', 
    'Temp_Humidity_Interaction', 'Hour_Load_Interaction'
]

# Ensure you have a 'models' directory
with open('models/feature_list.pkl', 'wb') as f:
    pickle.dump(correct_feature_list, f)

print("FINAL 'models/feature_list.pkl' created successfully.")
print("This list matches your saved models.")