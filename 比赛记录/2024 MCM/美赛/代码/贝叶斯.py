import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plt

# Load the dataset
file_path = "C:/Users/92579/Desktop/2024_MCM-ICM_Problems/2024_MCM-ICM_Problems/Wimbledon_featured_matches.csv"
df = pd.read_csv(file_path, nrows=301)  # Read only the first 301 rows

# Preprocessing
df['elapsed_time'] = pd.to_timedelta(df['elapsed_time'])
df.set_index('elapsed_time', inplace=True)

# Prepare the data in the required format for Prophet
df_prophet = df.reset_index().rename(columns={'elapsed_time': 'ds', 'rally_count': 'y'})

# Split the dataset into training and test sets (70-30 split)
train_size = int(len(df_prophet) * 0.7)
train, test = df_prophet[:train_size], df_prophet[train_size:]

# Create and fit the Prophet model
model = Prophet()
model.fit(train)

# Make future dataframe for forecasting
future = model.make_future_dataframe(periods=len(test), freq='D')

# Forecasting
forecast = model.predict(future)

# Plotting
fig = model.plot(forecast)
plt.xlabel('Elapsed Time')
plt.ylabel('Rally Count')
plt.show()
