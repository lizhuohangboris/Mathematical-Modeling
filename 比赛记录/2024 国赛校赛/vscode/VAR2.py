import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error

# Load data
training_data_path = 'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问训练数据.xlsx'
prediction_data_path = 'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问.xlsx'

# Read the data into DataFrames
training_data = pd.read_excel(training_data_path)
prediction_data = pd.read_excel(prediction_data_path)

# Set the index to '年份' for time series analysis
training_data.set_index('年份', inplace=True)
prediction_data.set_index('年份', inplace=True)

# Separate features and target variable
features = ['GDP（十亿）', '人口（百万人）', '钢铁产量（千吨）', '水泥（百万吨）', '民用汽车数量（千辆）', '煤炭消耗量（百万吨）', '原油消耗量', '天然气消耗量', '新能源消耗量']
target = '二氧化碳排放量（百万吨）'

# Create time series data including all variables
time_series_data = training_data[features + [target]]

# Split into training and testing sets (80% training, 20% testing)
train_size = int(len(time_series_data) * 0.8)
train_series, test_series = time_series_data[:train_size], time_series_data[train_size:]

# Train VAR model
var_model = VAR(train_series)
var_result = var_model.fit(maxlags=5)

# Forecast using the VAR model
lag_order = var_result.k_ar
forecast_input = train_series.values[-lag_order:]
forecast = var_result.forecast(y=forecast_input, steps=len(test_series))

# Convert forecast results to DataFrame
forecast_df = pd.DataFrame(forecast, index=test_series.index, columns=time_series_data.columns)

# Calculate mean squared error
mse_var = mean_squared_error(test_series[target], forecast_df[target])

# Forecast future CO2 emissions
future_forecast = var_result.forecast(y=time_series_data.values[-lag_order:], steps=len(prediction_data))
future_forecast_df = pd.DataFrame(future_forecast, index=prediction_data.index, columns=time_series_data.columns)

# Add forecast results to the prediction data
prediction_data['二氧化碳排放量（百万吨）'] = future_forecast_df[target].values

# Save the prediction results to a new Excel file
output_path_var = 'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问预测结果_VAR.xlsx'
prediction_data.to_excel(output_path_var, index=False)

output_path_var, mse_var
