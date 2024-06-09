import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Load the training data
data = pd.read_excel('D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问训练数据.xlsx')
data.set_index('年份', inplace=True)  # 使用实际的列名

# Renaming columns to match the variables used in the model
data.rename(columns={
    'GDP（十亿）': 'GDP',
    '人口（百万人）': 'Population',
    '二氧化碳排放量（百万吨）': 'CO2_Emissions',
    '钢铁产量（千吨）': 'Steel_Production',
    '水泥（百万吨）': 'Cement_Production',
    '民用汽车数量（千辆）': 'Vehicle_Numbers',
    '煤炭消耗量（百万吨）': 'Coal_Consumption',
    '原油消耗量': 'Oil_Consumption',
    '天然气消耗量': 'Gas_Consumption',
    '新能源消耗量': 'Renewable_Energy'
}, inplace=True)

# Selecting the necessary columns for modeling
variables = ['GDP', 'Population', 'Steel_Production', 'Cement_Production', 'Vehicle_Numbers', 'Coal_Consumption', 'Oil_Consumption', 'Gas_Consumption', 'Renewable_Energy']
CO2_emissions = data['CO2_Emissions']

# Fit the SARIMAX model (equivalent to ARIMA with exogenous variables)
model = SARIMAX(CO2_emissions, exog=data[variables], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
results = model.fit()

# Forecast future values
future_years = pd.DataFrame({
    '年份': [2030, 2060],
    'GDP': [data['GDP'].iloc[-1] * 1.5, data['GDP'].iloc[-1] * 2],  # Example projection
    'Population': [data['Population'].iloc[-1] * 1.05, data['Population'].iloc[-1] * 1.1],  # Example projection
    'Steel_Production': [data['Steel_Production'].iloc[-1] * 1.1, data['Steel_Production'].iloc[-1] * 1.2],  # Example projection
    'Cement_Production': [data['Cement_Production'].iloc[-1] * 1.1, data['Cement_Production'].iloc[-1] * 1.2],  # Example projection
    'Vehicle_Numbers': [data['Vehicle_Numbers'].iloc[-1] * 1.2, data['Vehicle_Numbers'].iloc[-1] * 1.5],  # Example projection
    'Coal_Consumption': [data['Coal_Consumption'].iloc[-1] * 0.9, data['Coal_Consumption'].iloc[-1] * 0.8],  # Example reduction
    'Oil_Consumption': [data['Oil_Consumption'].iloc[-1] * 1.1, data['Oil_Consumption'].iloc[-1] * 1.2],  # Example projection
    'Gas_Consumption': [data['Gas_Consumption'].iloc[-1] * 1.2, data['Gas_Consumption'].iloc[-1] * 1.5],  # Example projection
    'Renewable_Energy': [data['Renewable_Energy'].iloc[-1] * 2, data['Renewable_Energy'].iloc[-1] * 3]  # Example increase
})
future_years.set_index('年份', inplace=True)

forecast = results.get_forecast(steps=len(future_years), exog=future_years)
forecast_mean = forecast.predicted_mean

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(data.index, CO2_emissions, label='Observed CO2 Emissions')
plt.plot(future_years.index, forecast_mean, label='Forecasted CO2 Emissions', linestyle='--')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (Million Tons)')
plt.title('Forecasted CO2 Emissions for 2030 and 2060')
plt.legend()
plt.grid(True)
plt.show()

# Print the forecasted values
print(forecast_mean)
