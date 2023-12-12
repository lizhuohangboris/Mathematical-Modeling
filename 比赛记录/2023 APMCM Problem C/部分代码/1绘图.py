import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# data
data = {
    'Year': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'Sale Volume': [0.9, 1.3, 1.8, 7.5, 32.9, 50.2, 76.8, 124.7, 120.6, 132.3, 350.7, 385.1],
    'Per Capita Income': [14551, 16510, 18311, 20167, 21966, 23821, 25974, 28228, 30733, 32189, 35128, 36883],
    'Charging Piles': [0.68, 1.17, 2.3, 3.96, 5.78, 20.4, 44.6, 77.7, 121.9, 168.1, 261.7, 521],
    'Average Oil Price': [7.63, 7.83, 7.73, 7.51, 6.07, 5.95, 6.42, 7.22, 6.81, 5.71, 6.83, 7.61],
    'Government Budget': [0.7, 1.3, 2.1, 4.2, 27.8, 212.7, 375.6, 471.1, 274.3, 152.9, 194.8, 199],
    'NEV Energy Efficiency': [4, 4.1, 4.2, 4.5, 4.8, 5.1, 5.4, 5.7, 6, 6.3, 6.6, 6.9]
}

# Create DataFrame
df = pd.DataFrame(data)

# Scatter matrix
sns.pairplot(df)
plt.show()

# Thermal map
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()
