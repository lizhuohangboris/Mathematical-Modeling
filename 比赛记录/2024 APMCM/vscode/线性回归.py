import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# New Data (Years, Market Size, and other features)
data = {
    'Years': [2019, 2020, 2021, 2022, 2023],
    'Market_Size': [98.57, 111.9, 121, 133.7, 143.6],  # Global pet food market size in billion USD
    'Unit_Price': [2.26, 2.4, 2.55, 2.7, 2.79],
    'Online_Share': [4.2, 5.3, 6, 6.6, 7.4],  # Share of online sales in percentage
    'Offline_Share': [95.8, 94.7, 94, 93.4, 92.6],  # Share of offline sales in percentage
    'Growth_Percentage': [6.15, 11.72, 11.47, 15.69, 12.15],  # Market growth in percentage
    'Average_Price': [2.26, 2.4, 2.55, 2.7, 2.79]  # Average price (this seems to be the same as 'Unit_Price')
}

# Convert data into a pandas DataFrame
df = pd.DataFrame(data)

# Feature variables (independent variables) - predictors
X = df[['Unit_Price', 'Online_Share', 'Offline_Share', 'Growth_Percentage']]

# Target variables (dependent variables)
y_market_size = df['Market_Size']
y_unit_price = df['Unit_Price']
y_online_share = df['Online_Share']
y_offline_share = df['Offline_Share']
y_growth_percentage = df['Growth_Percentage']
y_average_price = df['Average_Price']

# Normalize the features using StandardScaler (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 1: Linear Regression for Market Size ---
market_size_model = LinearRegression()
market_size_model.fit(X_scaled, y_market_size)

# --- Step 2: Neural Network Model for other variables ---
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_dim=X_scaled.shape[1]),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer with 1 neuron
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train models for other variables
models = {}
for target, y in zip(['Unit_Price', 'Online_Share', 'Offline_Share', 'Growth_Percentage', 'Average_Price'],
                      [y_unit_price, y_online_share, y_offline_share, y_growth_percentage, y_average_price]):
    model = build_model()
    model.fit(X_scaled, y, epochs=100, batch_size=5, verbose=0)
    models[target] = model

# Future years data (hypothetical values for 2024, 2025, 2026)
future_years = [2024, 2025, 2026]
future_data = np.array([
    [2.85, 8.0, 91.0, 13.0],  # Hypothetical values for 2024
    [3.0, 8.5, 90.5, 12.5],  # Hypothetical values for 2025
    [3.1, 9.0, 90.0, 12.0]   # Hypothetical values for 2026
])

# Normalize the future data using the same scaler
future_data_scaled = scaler.transform(future_data)

# Predicting values for 2024, 2025, 2026 using the trained models
predictions = {}
predictions['Market_Size'] = market_size_model.predict(future_data_scaled).flatten()  # Linear Regression for Market Size
for target, model in models.items():
    predictions[target] = model.predict(future_data_scaled).flatten()

# Create predictions DataFrame for the future years
predictions_df = pd.DataFrame({
    'Year': future_years,
    'Predicted_Market_Size': predictions['Market_Size'],
    'Predicted_Unit_Price': predictions['Unit_Price'],
    'Predicted_Online_Share': predictions['Online_Share'],
    'Predicted_Offline_Share': predictions['Offline_Share'],
    'Predicted_Growth_Percentage': predictions['Growth_Percentage'],
    'Predicted_Average_Price': predictions['Average_Price']
})

# Print predictions table
print(predictions_df)

# Combine historical data with predicted data for the full range (2019-2026)
all_years = df['Years'].tolist() + future_years
all_market_size = df['Market_Size'].tolist() + list(predictions['Market_Size'])
all_unit_price = df['Unit_Price'].tolist() + list(predictions['Unit_Price'])
all_online_share = df['Online_Share'].tolist() + list(predictions['Online_Share'])
all_offline_share = df['Offline_Share'].tolist() + list(predictions['Offline_Share'])
all_growth_percentage = df['Growth_Percentage'].tolist() + list(predictions['Growth_Percentage'])
all_average_price = df['Average_Price'].tolist() + list(predictions['Average_Price'])

# Plotting all values: historical and predicted
plt.figure(figsize=(10, 6))

# Plot Market Size
plt.plot(all_years, all_market_size, label='Market Size', marker='o', color='tab:blue')

# Plot Unit Price
plt.plot(all_years, all_unit_price, label='Unit Price', marker='o', color='tab:orange')

# Plot Online Share
plt.plot(all_years, all_online_share, label='Online Share', marker='o', color='tab:green')

# Plot Offline Share
plt.plot(all_years, all_offline_share, label='Offline Share', marker='o', color='tab:red')

# Plot Growth Percentage
plt.plot(all_years, all_growth_percentage, label='Growth Percentage', marker='o', color='tab:purple')

# Plot Average Price
plt.plot(all_years, all_average_price, label='Average Price', marker='o', color='tab:brown')

# Labels and title
plt.xlabel('Year')
plt.ylabel('Values')
plt.title('Historical and Predicted Pet Food Market Data')
plt.legend()
plt.grid(True)

# Show plot
plt.show()
