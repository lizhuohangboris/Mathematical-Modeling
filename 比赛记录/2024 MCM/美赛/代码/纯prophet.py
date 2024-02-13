import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load data from '数据处理.csv' with specific columns
file_path = "c:/Users/92579/Documents/GitHub/Mathematical-Modeling/比赛记录/2024 MCM/美赛/2024_MCM-ICM_Problems/2024_MCM-ICM_Problems/数据处理.csv"
columns_to_read = ['elapsed_time', 'point_victor']
df = pd.read_csv(file_path, usecols=columns_to_read)

# Convert 'elapsed_time' to datetime
df['elapsed_time'] = pd.to_datetime(df['elapsed_time'])

# Prophet model for time series prediction
prophet_model = Prophet()

# Rename columns for Prophet
train_prophet = df[['elapsed_time', 'point_victor']]
train_prophet = train_prophet.rename(columns={'elapsed_time': 'ds', 'point_victor': 'y'})

# Fit the Prophet model
prophet_model.fit(train_prophet)

# Create a dataframe with future timestamps for prediction
future = pd.DataFrame(pd.date_range(start=df['elapsed_time'].min(), end=df['elapsed_time'].max(), freq='1H'), columns=['ds'])

# Make predictions on the entire dataset
prophet_predictions = prophet_model.predict(future)

# Plot the time series of predicted probabilities for Prophet
plt.figure(figsize=(15, 8))

# Subplot 1: Prophet Predicted Probability and Original Data Points
plt.plot(df['elapsed_time'], df['point_victor'], label='Original Data Points', linestyle='', marker='o', color='black', markersize=3, alpha=0.7)
plt.plot(prophet_predictions['ds'], prophet_predictions['yhat'], label='Prophet Predicted Probability', linestyle='-', color='salmon')
plt.fill_between(prophet_predictions['ds'], prophet_predictions['yhat_lower'], prophet_predictions['yhat_upper'], color='salmon', alpha=0.2)
plt.xlabel('Elapsed Time')
plt.ylabel('Point Victor (0 or 1)')
plt.title('Prophet - Time Series of Original Data and Predicted Probability')
plt.legend()

plt.tight_layout()
plt.show()

# 获取Prophet模型的中间量
components = prophet_model.predict(future)
trend_component = components['trend']
seasonal_component = components['yearly']

# 输出趋势和季节性中间量
print("Trend Component:")
print(trend_component.head())

print("\nSeasonal Component:")
print(seasonal_component.head())
