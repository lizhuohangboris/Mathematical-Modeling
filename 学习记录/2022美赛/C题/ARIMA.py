import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# 读取CSV文件
file_path = "C:/Users/92579/Documents/GitHub/Mathematical-Modeling/学习记录/2022美赛/C题/2022_Problem_C_DATA/BCHAIN-MKPRU.csv"
df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# 数据预处理
df = df.dropna()  # 删除缺失值
df = df.resample('D').mean()  # 将数据按天重采样，取均值

# 可视化数据
plt.figure(figsize=(10, 6))
plt.plot(df['Value'])
plt.title('Bitcoin Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# 进行稳定性检验（Augmented Dickey-Fuller Test）
result_adf = adfuller(df['Value'])
print(f'ADF Statistic: {result_adf[0]}')
print(f'p-value: {result_adf[1]}')

# 如果时间序列不稳定，进行差分操作
if result_adf[1] > 0.05:
    df['Value'] = df['Value'].diff()
    df = df.dropna()

# 再次进行可视化
plt.figure(figsize=(10, 6))
plt.plot(df['Value'])
plt.title('Differenced Bitcoin Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price Difference')
plt.show()

# 根据ACF和PACF图确定p和q的值
plot_acf(df['Value'], lags=20)
plt.show()
plot_pacf(df['Value'], lags=20)
plt.show()

# 根据ACF和PACF图确定p和q的值
p = 2  # 根据图示确定
d = 1  # 一阶差分
q = 1  # 根据图示确定

# 拟合ARIMA模型
model = ARIMA(df['Value'], order=(p, d, q))
result = model.fit()

# 打印模型的参数
print(result.summary())

# 进行白噪声检验
residuals = result.resid
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(residuals, ax=ax[0])
plot_pacf(residuals, ax=ax[1])
plt.show()

# 进行模型评估和预测
future_days = 3  # 设置预测未来几天的数据
forecast = result.forecast(steps=future_days)

# 打印预测结果
print("Forecasted Values:")
print(forecast)

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(df['Value'], label='Historical Data')
plt.plot(pd.date_range(df.index[-1], periods=future_days + 1)[1:], forecast, label='Forecast', color='red')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
