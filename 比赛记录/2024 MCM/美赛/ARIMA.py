import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt

# Load the dataset
file_path = "C:/Users/92579/Desktop/2024_MCM-ICM_Problems/2024_MCM-ICM_Problems/Wimbledon_featured_matches.csv"
df = pd.read_csv(file_path, nrows=301)  # Read only the first 301 rows

# Preprocessing
df['elapsed_time'] = pd.to_timedelta(df['elapsed_time'])
df.set_index('elapsed_time', inplace=True)

# Time Series Decomposition (Optional)
# You can decompose the time series into trend, seasonality, and residuals if needed.

# Check for stationarity and apply differencing if needed
# For example, df['column'] = df['column'].diff()

# Split the dataset into training and test sets (70-30 split)
train_size = int(len(df) * 0.7)
train, test = df[:train_size], df[train_size:]

# Identification of ARIMA Parameters (Replace with your values)
p, d, q = 1, 1, 1  # Adjust these values based on your analysis

# Model fitting
model = ARIMA(train['rally_count'], order=(p, d, q))
fit_model = model.fit()

# Forecasting
forecast_steps = len(test)  # Forecast for the length of the test set
forecast = fit_model.get_forecast(steps=forecast_steps)
forecast_values = forecast.predicted_mean

# Plotting
plt.plot(train['rally_count'], label='Training Data')
plt.plot(test['rally_count'], label='Test Data')
plt.plot(test.index, forecast_values, label='Forecast', linestyle='dashed')
plt.xlabel('Elapsed Time')
plt.ylabel('Rally Count')
plt.legend()
plt.show()
