# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 加载Excel文件
file_path = 'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# 处理缺失值，通过线性插值填补二氧化碳排放量列中的缺失值
data['二氧化碳排放量（百万吨）'] = data['二氧化碳排放量（百万吨）'].interpolate(method='linear')

# 如果插值后仍有缺失值，使用列的均值填补
data['二氧化碳排放量（百万吨）'] = data['二氧化碳排放量（百万吨）'].fillna(data['二氧化碳排放量（百万吨）'].mean())

# 准备训练数据，自变量包括多个特征
X = data[['GDP（十亿）', '人口（百万人）', '钢铁产量（千吨）', '水泥（百万吨）', '民用汽车数量（千辆）',
          '煤炭消耗量（百万吨）', '原油消耗量', '天然气消耗量', '新能源消耗量']]
y = data['二氧化碳排放量（百万吨）']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 在测试集上进行预测并计算均方根误差（RMSE）
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'测试集的均方根误差（RMSE）：{rmse:.2f}')

# 构造未来年份的数据框，并设置未来年份的自变量（假设这些变量将以某种方式增长或保持不变）
future_years = np.arange(2023, 2061)
# 这里假设未来年份的自变量根据历史数据的平均增长率增长（可以根据实际情况调整）
future_data = pd.DataFrame({
    'GDP（十亿）': np.linspace(data['GDP（十亿）'].iloc[-1], data['GDP（十亿）'].iloc[-1] * 1.5, len(future_years)),
    '人口（百万人）': np.linspace(data['人口（百万人）'].iloc[-1], data['人口（百万人）'].iloc[-1] * 1.1, len(future_years)),
    '钢铁产量（千吨）': np.linspace(data['钢铁产量（千吨）'].iloc[-1], data['钢铁产量（千吨）'].iloc[-1] * 1.2, len(future_years)),
    '水泥（百万吨）': np.linspace(data['水泥（百万吨）'].iloc[-1], data['水泥（百万吨）'].iloc[-1] * 1.2, len(future_years)),
    '民用汽车数量（千辆）': np.linspace(data['民用汽车数量（千辆）'].iloc[-1], data['民用汽车数量（千辆）'].iloc[-1] * 1.3, len(future_years)),
    '煤炭消耗量（百万吨）': np.linspace(data['煤炭消耗量（百万吨）'].iloc[-1], data['煤炭消耗量（百万吨）'].iloc[-1] * 1.1, len(future_years)),
    '原油消耗量': np.linspace(data['原油消耗量'].iloc[-1], data['原油消耗量'].iloc[-1] * 1.1, len(future_years)),
    '天然气消耗量': np.linspace(data['天然气消耗量'].iloc[-1], data['天然气消耗量'].iloc[-1] * 1.1, len(future_years)),
    '新能源消耗量': np.linspace(data['新能源消耗量'].iloc[-1], data['新能源消耗量'].iloc[-1] * 1.5, len(future_years)),
    '年份': future_years
})

# 使用模型预测未来年份的二氧化碳排放量
future_X = future_data.drop(columns=['年份'])
future_predictions = model.predict(future_X)

# 将预测结果存储在DataFrame中
future_df = pd.DataFrame({'年份': future_years, '预测二氧化碳排放量（百万吨）': future_predictions})

# 显示未来预测结果的表格
print("未来二氧化碳排放量预测结果：")
print(future_df)

# 绘制历史数据和未来预测的图表
plt.figure(figsize=(12, 6))
plt.plot(data['年份'], data['二氧化碳排放量（百万吨）'], label='历史数据')
plt.plot(future_df['年份'], future_df['预测二氧化碳排放量（百万吨）'], label='未来预测', linestyle='--')
plt.xlabel('年份')
plt.ylabel('二氧化碳排放量（百万吨）')
plt.title('2023年至2060年二氧化碳排放量预测')
plt.legend()
plt.grid(True)
plt.show()

# 打印线性回归方程
coefficients = model.coef_
intercept = model.intercept_
equation = f'二氧化碳排放量（百万吨） = {intercept:.2f} + ' + ' + '.join([f'{coeff:.2f} * {col}' for coeff, col in zip(coefficients, X.columns)])
print('线性回归方程：', equation)
