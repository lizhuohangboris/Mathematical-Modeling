# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# 加载Excel文件
file_path = 'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# 处理缺失值，通过线性插值填补二氧化碳排放量列中的缺失值
data['二氧化碳排放量（百万吨）'] = data['二氧化碳排放量（百万吨）'].interpolate(method='linear')

# 如果插值后仍有缺失值，使用列的均值填补
data['二氧化碳排放量（百万吨）'] = data['二氧化碳排放量（百万吨）'].fillna(data['二氧化碳排放量（百万吨）'].mean())

# 使用年份作为索引
data.set_index('年份', inplace=True)

# 拆分数据为训练集和测试集
train = data['二氧化碳排放量（百万吨）'][:int(0.8*len(data))]
test = data['二氧化碳排放量（百万吨）'][int(0.8*len(data)):]

# 创建并训练ARIMA模型
model = ARIMA(train, order=(5, 1, 0))  # 这里order可以根据实际情况调整
model_fit = model.fit()

# 在测试集上进行预测
predictions = model_fit.forecast(steps=len(test))
test_mse = mean_squared_error(test, predictions)
print(f'测试集的均方根误差（RMSE）：{np.sqrt(test_mse):.2f}')

# 预测未来年份的二氧化碳排放量
future_years = np.arange(2023, 2061)
future_predictions = model_fit.forecast(steps=len(future_years))

# 将预测结果存储在DataFrame中
future_df = pd.DataFrame({'年份': future_years, '预测二氧化碳排放量（百万吨）': future_predictions})
future_df.set_index('年份', inplace=True)

# 显示未来预测结果的表格
print("未来二氧化碳排放量预测结果：")
print(future_df)

# 绘制历史数据和未来预测的图表
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['二氧化碳排放量（百万吨）'], label='历史数据')
plt.plot(future_df.index, future_df['预测二氧化碳排放量（百万吨）'], label='未来预测', linestyle='--')
plt.xlabel('年份')
plt.ylabel('二氧化碳排放量（百万吨）')
plt.title('2023年至2060年二氧化碳排放量预测')
plt.legend()
plt.grid(True)
plt.show()
