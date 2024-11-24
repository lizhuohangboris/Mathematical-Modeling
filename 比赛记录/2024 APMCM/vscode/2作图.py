import matplotlib.pyplot as plt
import numpy as np

# Historical Data (2019-2023)
historical_years = [2019, 2020, 2021, 2022, 2023]
historical_market_size = [98.57, 111.9, 121, 133.7, 143.6]  # Global pet food market size in billion USD
historical_unit_price = [2.26, 2.4, 2.55, 2.7, 2.79]
historical_online_share = [4.2, 5.3, 6, 6.6, 7.4]  # Share of online sales in percentage
historical_offline_share = [95.8, 94.7, 94, 93.4, 92.6]  # Share of offline sales in percentage
historical_growth_percentage = [6.15, 11.72, 11.47, 15.69, 12.15]  # Market growth in percentage

# Predicted Data (2024-2026)
predicted_years = [2024, 2025, 2026]
predicted_market_size = [148.56, 155.23, 160.21]
predicted_unit_price = [2.85, 3.00, 3.10]
predicted_online_share = [8.0, 8.5, 9.0]
predicted_offline_share = [91.0, 90.5, 90.0]
predicted_growth_percentage = [13.0, 12.5, 12.0]

# Create the plots with subplots for better organization
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Pie chart for market share (2023 vs 2026)
axs[0, 0].pie([historical_online_share[-1], historical_offline_share[-1]], 
              labels=['Online', 'Offline'],
              autopct='%1.1f%%', startangle=90, colors=['tab:green', 'tab:red'])
axs[0, 0].set_title('Market Share (2023 vs 2026)')

# Bar chart for Growth Percentage (2019-2026)
historical_years_expanded = historical_years + predicted_years  # Combine historical and predicted years
growth_percentage_expanded = historical_growth_percentage + predicted_growth_percentage  # Combine growth percentages

bar_width = 0.35
index = np.arange(len(historical_years) + len(predicted_years))  # Create a longer index

# Plot historical growth percentage
axs[0, 1].bar(index[:len(historical_years)], historical_growth_percentage, bar_width, label='Historical', color='tab:purple')
# Plot predicted growth percentage
axs[0, 1].bar(index[len(historical_years):], predicted_growth_percentage, bar_width, label='Predicted', color='tab:blue')

axs[0, 1].set_xticks(index)
axs[0, 1].set_xticklabels(historical_years + predicted_years)
axs[0, 1].set_xlabel('Year')
axs[0, 1].set_ylabel('Growth Percentage (%)')
axs[0, 1].set_title('Market Growth Percentage (2019-2026)')
axs[0, 1].legend()

# Add data labels for Growth Percentage Bar Chart
for i, v in enumerate(historical_growth_percentage):
    axs[0, 1].text(i - bar_width / 2, v + 0.5, f'{v:.1f}%', ha='center', color='white')
for i, v in enumerate(predicted_growth_percentage):
    axs[0, 1].text(len(historical_years) + i + bar_width / 2, v + 0.5, f'{v:.1f}%', ha='center', color='white')

# Line plot for market size (historical and predicted)
axs[1, 0].plot(historical_years, historical_market_size, label='Market Size (Billion USD)', marker='o', color='tab:blue')
axs[1, 0].plot(predicted_years, predicted_market_size, '--', marker='o', color='tab:blue')
axs[1, 0].set_xlabel('Year')
axs[1, 0].set_ylabel('Market Size (Billion USD)')
axs[1, 0].set_title('Market Size Over Time')
axs[1, 0].legend()

# Add data labels for Market Size Line Plot
for i, year in enumerate(historical_years):
    axs[1, 0].text(year, historical_market_size[i] + 1, f'{historical_market_size[i]:.2f}', ha='center', color='tab:blue')
for i, year in enumerate(predicted_years):
    axs[1, 0].text(year, predicted_market_size[i] + 1, f'{predicted_market_size[i]:.2f}', ha='center', color='tab:blue')

# Line plot for unit price (historical and predicted)
axs[1, 1].plot(historical_years, historical_unit_price, label='Unit Price (USD)', marker='o', color='tab:orange')
axs[1, 1].plot(predicted_years, predicted_unit_price, '--', marker='o', color='tab:orange')
axs[1, 1].set_xlabel('Year')
axs[1, 1].set_ylabel('Unit Price (USD)')
axs[1, 1].set_title('Unit Price Over Time')
axs[1, 1].legend()

# Add data labels for Unit Price Line Plot
for i, year in enumerate(historical_years):
    axs[1, 1].text(year, historical_unit_price[i] + 0.05, f'{historical_unit_price[i]:.2f}', ha='center', color='tab:orange')
for i, year in enumerate(predicted_years):
    axs[1, 1].text(year, predicted_unit_price[i] + 0.05, f'{predicted_unit_price[i]:.2f}', ha='center', color='tab:orange')

# Show the plots
plt.tight_layout()
plt.show()
