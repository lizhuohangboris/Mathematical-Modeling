# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 加载Excel文件
file_path = 'D://课程/比赛/数学建模/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# 处理缺失值，通过线性插值填补
data.interpolate(method='linear', inplace=True)

# 如果插值后仍有缺失值，使用列的均值填补
data.fillna(data.mean(), inplace=True)

# 准备训练数据，自变量包括多个特征
X = data[['年份', 'GDP(十亿)', '人口（百万人）', '钢铁产量（千吨）', '水泥（百万吨）', '民用汽车数量（千辆）',
          '煤炭消耗量（百万吨）', '原油消耗量', '天然气消耗量', '新能源消耗量']]
y = data['二氧化碳排放量（百万吨）']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 在测试集上进行预测并计算均方根误差（RMSE）
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'测试集的均方根误差（RMSE）：{rmse:.2f}')

# 构造未来年份的数据框，并设置未来年份的自变量
future_years = np.arange(2023, 2061)
future_data = pd.DataFrame({
    '年份': future_years,
    'GDP（十亿）': np.linspace(data['GDP（十亿）'].iloc[-1], data['GDP（十亿）'].iloc[-1] * 1.5, len(future_years)),
    '人口（百万人）': np.linspace(data['人口（百万人）'].iloc[-1], data['人口（百万人）'].iloc[-1] * 1.1, len(future_years)),
    '钢铁产量（千吨）': np.linspace(data['钢铁产量（千吨）'].iloc[-1], data['钢铁产量（千吨）'].iloc[-1] * 1.2, len(future_years)),
    '水泥（百万吨）': np.linspace(data['水泥（百万吨）'].iloc[-1], data['水泥（百万吨）'].iloc[-1] * 1.2, len(future_years)),
    '民用汽车数量（千辆）': np.linspace(data['民用汽车数量（千辆）'].iloc[-1], data['民用汽车数量（千辆）'].iloc[-1] * 1.3, len(future_years)),
    '煤炭消耗量（百万吨）': np.linspace(data['煤炭消耗量（百万吨）'].iloc[-1], data['煤炭消耗量（百万吨）'].iloc[-1] * 1.1, len(future_years)),
    '原油消耗量': np.linspace(data['原油消耗量'].iloc[-1], data['原油消耗量'].iloc[-1] * 1.1, len(future_years)),
    '天然气消耗量': np.linspace(data['天然气消耗量'].iloc[-1], data['天然气消耗量'].iloc[-1] * 1.1, len(future_years)),
    '新能源消耗量': np.linspace(data['新能源消耗量'].iloc[-1], data['新能源消耗量'].iloc[-1] * 1.5, len(future_years))
})

# 使用模型预测未来年份的二氧化碳排放量
future_predictions = model.predict(future_data)

# 将预测结果存储在DataFrame中
future_df = pd.DataFrame({'年份': future_years, '预测二氧化碳排放量（百万吨）': future_predictions})

# 提取2030年和2060年的预测值
pred_2030 = future_df[future_df['年份'] == 2030]['预测二氧化碳排放量（百万吨）'].values[0]
pred_2060 = future_df[future_df['年份'] == 2060]['预测二氧化碳排放量（百万吨）'].values[0]

# 输出2030年和2060年的预测值
print(f'2030年的预测二氧化碳排放量（百万吨）：{pred_2030:.2f}')
print(f'2060年的预测二氧化碳排放量（百万吨）：{pred_2060:.2f}')
