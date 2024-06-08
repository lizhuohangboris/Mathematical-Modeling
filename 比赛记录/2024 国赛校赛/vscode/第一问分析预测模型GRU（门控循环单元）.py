import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

# 读取数据
file_path = r'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/数据(2).xlsx'
data = pd.read_excel(file_path)

# 提取年份
years = data['年份'].values
# 选择需要的特征列
feature_cols = ['GDP（十亿）', '人口（百万人）', '能源消耗', '钢铁产量（千吨）', '水泥（百万吨）', '民用汽车数量（千辆）',
                '煤炭消耗量（百万吨）', '原油消耗量', '天然气消耗量', '新能源消耗量']
target_col = '二氧化碳排放量（百万吨）'
selected_data = data[feature_cols + [target_col]]

# 数据预处理
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(selected_data)

# 划分训练集和测试集
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps, -1])
    return np.array(X), np.array(y)

time_steps = 3  # 设置时间步长
X_train, y_train = create_dataset(train_data, time_steps)
X_test, y_test = create_dataset(test_data, time_steps)

# 构建GRU模型
model = Sequential([
    GRU(50, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 拟合模型
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# 使用模型进行预测
y_pred = model.predict(X_test)

# 计算预测结果与实际结果之间的均方误差
mse = mean_squared_error(y_test, y_pred)
print('均方误差 (MSE):', mse)

# 计算预测结果与实际结果之间的均方根误差
rmse = np.sqrt(mse)
print('均方根误差 (RMSE):', rmse)

# 计算预测结果与实际结果之间的平均绝对误差
mae = mean_absolute_error(y_test, y_pred)
print('平均绝对误差 (MAE):', mae)

# 创建一个新的缩放器对象
scaler_y = MinMaxScaler()

# 拟合缩放器对象并进行缩放
scaler_y.fit(selected_data[[target_col]])

# 将预测值进行反向缩放
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1))

# 将真实值进行反向缩放
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# 将年份转换为 datetime 对象
years_int = years.astype(int)
dates = [datetime(year, 1, 1) for year in years_int]

# 选择从1970年开始的数据索引
start_index = np.where(years == 1970)[0][0]

# 可视化真实值和预测值
plt.figure(figsize=(10, 6))
plt.plot(dates[start_index:start_index+len(y_test)], y_test_original, label='Actual (Test)', color='blue', linestyle='-')
plt.plot(dates[start_index:start_index+len(y_pred)], y_pred_original, label='Predicted (Test)', color='orange', linestyle='--')
plt.plot(dates[:len(y_train)], y_train_original, label='Actual (Train)', color='green', linestyle='-')
plt.title('Actual vs Predicted CO2 Emissions Over Time')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (million tons)')
plt.legend()
plt.grid(True)
plt.show()
