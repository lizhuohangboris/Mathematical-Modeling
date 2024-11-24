import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Data (Years, Market Size, and other features)
data = {
    'Years': [2019, 2020, 2021, 2022, 2023],
    'Market_Size': [98.57, 111.9, 121, 133.7, 143.6],  # Global pet food market size in billion USD
    'Unit_Price': [2.26, 2.4, 2.55, 2.7, 2.79],
    'Online_Share': [4.2, 5.3, 6, 6.6, 7.4],  # Share of online sales in percentage
    'Offline_Share': [95.8, 94.7, 94, 93.4, 92.6],  # Share of offline sales in percentage
    'Growth_Percentage': [6.15, 11.72, 11.47, 15.69, 12.15]  # Market growth in percentage
}

# Convert data into a pandas DataFrame
df = pd.DataFrame(data)

# Feature variables (independent variables) - predictors
X = df[['Unit_Price', 'Online_Share', 'Offline_Share', 'Growth_Percentage']]

# Target variables (dependent variables) - We will predict Market_Size, Unit_Price, etc.
y_market_size = df['Market_Size']
y_unit_price = df['Unit_Price']
y_online_share = df['Online_Share']
y_offline_share = df['Offline_Share']
y_growth_percentage = df['Growth_Percentage']

# Normalize the features using StandardScaler (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the Linear Regression model
model = LinearRegression()

# Train models for each target variable (Market_Size, Unit_Price, Online_Share, Offline_Share, Growth_Percentage)
model.fit(X_scaled, y_market_size)
model.fit(X_scaled, y_unit_price)
model.fit(X_scaled, y_online_share)
model.fit(X_scaled, y_offline_share)
model.fit(X_scaled, y_growth_percentage)

# Future years data (hypothetical values for 2024, 2025, 2026)
future_years = [2024, 2025, 2026]
future_data = np.array([
    [2.85, 8.0, 91.0, 13.0],  # Hypothetical values for 2024
    [3.0, 8.5, 90.5, 12.5],  # Hypothetical values for 2025
    [3.1, 9.0, 90.0, 12.0]   # Hypothetical values for 2026
])

# Normalize the future data using the same scaler
future_data_scaled = scaler.transform(future_data)

# Predicting values for 2024, 2025, 2026
pred_market_size = model.predict(future_data_scaled)
pred_unit_price = model.predict(future_data_scaled)
pred_online_share = model.predict(future_data_scaled)
pred_offline_share = model.predict(future_data_scaled)
pred_growth_percentage = model.predict(future_data_scaled)

# Create predictions DataFrame for the future years
predictions_df = pd.DataFrame({
    'Year': future_years,
    'Predicted_Market_Size': pred_market_size,
    'Predicted_Unit_Price': pred_unit_price,
    'Predicted_Online_Share': pred_online_share,
    'Predicted_Offline_Share': pred_offline_share,
    'Predicted_Growth_Percentage': pred_growth_percentage
})

# Print predictions table
print(predictions_df)

# Combine historical data with predicted data for the full range (2019-2026)
all_years = df['Years'].tolist() + future_years
all_market_size = df['Market_Size'].tolist() + list(pred_market_size)
all_unit_price = df['Unit_Price'].tolist() + list(pred_unit_price)
all_online_share = df['Online_Share'].tolist() + list(pred_online_share)
all_offline_share = df['Offline_Share'].tolist() + list(pred_offline_share)
all_growth_percentage = df['Growth_Percentage'].tolist() + list(pred_growth_percentage)

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

# Labels and title
plt.xlabel('Year')
plt.ylabel('Values')
plt.title('Historical and Predicted Pet Food Market Data')
plt.legend()
plt.grid(True)

# Show plot
plt.show()
