
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the Excel file
file_path = 'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/data_energe.xlsx'  # Adjust the file path as needed
data = pd.read_excel(file_path)

# Convert '年份' to datetime and set as index
data['年份'] = pd.to_datetime(data['年份'])
data.set_index('年份', inplace=True)

# Define the forecast range properly
forecast_range = pd.date_range(start='2022-01-01', end='2072-12-01', freq='YS')

# Define a function to apply ARIMA and forecast the given column
def forecast_arima(data, column, forecast_range):
    # Fit ARIMA model
    model = ARIMA(data[column], order=(5,1,0))
    model_fit = model.fit()
    
    # Forecast
    forecast = model_fit.get_forecast(steps=len(forecast_range))
    forecast_values = forecast.predicted_mean
    forecast_values.index = forecast_range
    
    return forecast_values

# Forecast for each column from 2022 to 2072
forecasts = {}
for column in data.columns:
    forecasts[column] = forecast_arima(data, column, forecast_range)

# Sum the forecasts to get the total energy generation
total_forecast = sum(forecasts.values())

# Create a DataFrame for visualization
forecast_df = pd.DataFrame(forecasts)
forecast_df['总量'] = total_forecast

# Plot the results
plt.figure(figsize=(12, 8))
for column in forecast_df.columns:
    plt.plot(forecast_df.index, forecast_df[column], label=column)
plt.xlabel('年份')
plt.ylabel('发电量 (亿千瓦时)')
plt.title('2022-2072年各能源发电量预测')
plt.legend()
plt.show()

import ace_tools as tools
tools.display_dataframe_to_user(name="Forecasted Energy Generation (2022-2072)", dataframe=forecast_df)
