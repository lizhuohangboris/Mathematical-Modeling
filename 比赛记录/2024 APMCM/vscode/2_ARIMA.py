import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# 构建数据
data = {
    'Year': [2019, 2020, 2021, 2022, 2023],
    'US_Cats': [9420, 6500, 9420, 7380, 7380],
    'US_Dogs': [8970, 8500, 8970, 8970, 8010],
    'France_Cats': [1300, 1490, 1510, 1490, 1660],
    'France_Dogs': [740, 775, 750, 760, 990],
    'Germany_Cats': [1470, 1570, 1670, 1520, 1570],
    'Germany_Dogs': [1010, 1070, 1030, 1060, 1050]
}

df = pd.DataFrame(data)

# 将年份设置为时间索引
df.set_index('Year', inplace=True)

# 使用auto_arima选择最佳模型
def arima_forecast_auto(series, steps=3):
    model = auto_arima(series, seasonal=False, stepwise=True, trace=True)
    forecast = model.predict(n_periods=steps)
    return forecast

# 预测未来3年
us_cats_forecast = arima_forecast_auto(df['US_Cats'])
us_dogs_forecast = arima_forecast_auto(df['US_Dogs'])
france_cats_forecast = arima_forecast_auto(df['France_Cats'])
france_dogs_forecast = arima_forecast_auto(df['France_Dogs'])
germany_cats_forecast = arima_forecast_auto(df['Germany_Cats'])
germany_dogs_forecast = arima_forecast_auto(df['Germany_Dogs'])

# 输出预测结果
forecast_data = {
    'US_Cats': us_cats_forecast,
    'US_Dogs': us_dogs_forecast,
    'France_Cats': france_cats_forecast,
    'France_Dogs': france_dogs_forecast,
    'Germany_Cats': germany_cats_forecast,
    'Germany_Dogs': germany_dogs_forecast
}

forecast_df = pd.DataFrame(forecast_data, index=[2024, 2025, 2026])
print(forecast_df)

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['US_Cats'], label='US Cats', marker='o')
plt.plot([2024, 2025, 2026], us_cats_forecast, label='US Cats Forecast', linestyle='--')
plt.plot(df.index, df['US_Dogs'], label='US Dogs', marker='o')
plt.plot([2024, 2025, 2026], us_dogs_forecast, label='US Dogs Forecast', linestyle='--')
plt.legend()
plt.title('Pet Population Forecast (US)')
plt.xlabel('Year')
plt.ylabel('Population')
plt.show()
