import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure font settings for English
rcParams['font.sans-serif'] = ['Arial']  # Use Arial for English font
rcParams['axes.unicode_minus'] = False  # Ensure minus signs display correctly

# Data
years = [2019, 2020, 2021, 2022, 2023]
cat_population = [4412, 4862, 5806, 6536, 6980]  # in 10k
dog_population = [5503, 5222, 5429, 5119, 5175]  # in 10k
market_size = [2191, 2259, 2733, 3069, 3264]  # in 100 million yuan
food_market_size = [116.4, 138.2, 155.4, 173.2, 190]  # in 100 million yuan

# Plotting
plt.figure(figsize=(12, 6))

# Plotting the line charts for cat and dog populations
plt.plot(years, cat_population, marker='o', label='Cat Population (10k)', linewidth=2)
plt.plot(years, dog_population, marker='o', label='Dog Population (10k)', linewidth=2)

# Plotting the bar charts for market size
bars_market = plt.bar(
    [x - 0.2 for x in years],
    market_size,
    alpha=0.6,
    label='Pet Industry Market Size (100M yuan)',
    color='gray',
    width=0.4
)
bars_food = plt.bar(
    [x + 0.2 for x in years],
    food_market_size,
    alpha=0.6,
    label='Pet Food Market Size (100M yuan)',
    color='green',
    width=0.4
)

# Add data labels on the bars
for bar in bars_market:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 50, f'{int(height)}', ha='center', fontsize=10)

for bar in bars_food:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 5, f'{height:.1f}', ha='center', fontsize=10)

# Add data labels on the line charts for cats and dogs
for x, y in zip(years, cat_population):
    plt.text(x, y + 100, f'{y}', ha='center', fontsize=10, color='blue')
for x, y in zip(years, dog_population):
    plt.text(x, y + 100, f'{y}', ha='center', fontsize=10, color='orange')

# Add title and axis labels
plt.title(' ', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Population (10k) / Market Size (100M yuan)', fontsize=12)
plt.xticks(years)
plt.legend(fontsize=12)
plt.grid(alpha=0.5, linestyle='--')

# Show plot
plt.tight_layout()
plt.show()
