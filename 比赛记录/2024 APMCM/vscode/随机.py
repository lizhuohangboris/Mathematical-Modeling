import matplotlib.pyplot as plt
import numpy as np

# Data
years = [2019, 2020, 2021, 2022, 2023]
production_values = [440.7, 727.3, 1554, 1508, 2793]  # in CNY
export_values = [21.4, 9.8, 12.2, 24.7, 39.6]  # in USD

# Bar width
bar_width = 0.35

# X locations for the bars
x = np.arange(len(years))

# Create bars
fig, ax1 = plt.subplots()

# Bar chart for production values
bars1 = ax1.bar(x - bar_width / 2, production_values, bar_width, label='Production (CNY)', color='navajowhite')

# Add second y-axis for export values (USD)
ax2 = ax1.twinx()
bars2 = ax2.bar(x + bar_width / 2, export_values, bar_width, label='Exports (USD)', color='mediumseagreen')

# Labeling
ax1.set_xlabel('Year')
ax1.set_ylabel('Production Value (CNY)', color='g')
ax2.set_ylabel('Export Value (USD)', color='g')
ax1.set_title('China’s Pet Food Production and Export Values (2019-2023)')
ax1.set_xticks(x)
ax1.set_xticklabels(years)

# Extend the y-axes slightly
ax1.set_ylim(0, max(production_values) * 1.1)  # Extend production y-axis by 10%
ax2.set_ylim(0, max(export_values) * 1.2)     # Extend export y-axis by 20%

# Add data labels on top of the bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval + 50, round(yval, 2), ha='center', va='bottom', color='black')

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, yval + 1, round(yval, 2), ha='center', va='bottom', color='black')

# Adjust the legends to fit within the plot area
ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.93))

# Ensure the plot does not get clipped
plt.tight_layout()

# Show plot
plt.show()
