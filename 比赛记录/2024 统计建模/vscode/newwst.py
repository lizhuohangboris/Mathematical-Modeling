import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor




# 读取数据
data = pd.read_excel("D:/Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 统计建模/随机森林.xlsx")

# 将"Month"列转换为日期类型
data['Month'] = pd.to_datetime(data['Month'], format='%b-%y')

# 将"Month"列拆分为年和月，并添加这两列作为特征
data['Year'] = data['Month'].dt.year
data['Month'] = data['Month'].dt.month

# 确定特征和目标列
X = data.drop(["AQI"], axis=1)
y = data["AQI"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林模型
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 拟合随机森林模型
model_rf.fit(X_train, y_train)

# 使用随机森林模型对训练集和测试集进行预测
y_train_pred_rf = model_rf.predict(X_train)
y_test_pred_rf = model_rf.predict(X_test)
# 使用随机森林模型预测训练集和测试集的AQI值
y_train_pred_rf = model_rf.predict(X_train)
y_test_pred_rf = model_rf.predict(X_test)

# 将随机森林的预测结果添加到原始数据集中
X_train['RF_Prediction'] = y_train_pred_rf
X_test['RF_Prediction'] = y_test_pred_rf

# 数据缩放
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将数据重塑为 LSTM 所需的形状 (samples, time steps, features)
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# LSTM 模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 训练 LSTM 模型
model.fit(X_train_reshaped, y_train, epochs=100, verbose=0)

# 使用 LSTM 模型进行预测
y_train_pred_lstm = model.predict(X_train_reshaped)
y_test_pred_lstm = model.predict(X_test_reshaped)

# 计算模型在训练集和测试集上的R^2分数
train_score_lstm = model.evaluate(X_train_reshaped, y_train, verbose=0)
test_score_lstm = model.evaluate(X_test_reshaped, y_test, verbose=0)
print("LSTM Train MSE Score:", train_score_lstm)
print("LSTM Test MSE Score:", test_score_lstm)

# 绘制训练集真实值与预测值对比图
plt.figure(figsize=(10, 6))
plt.plot(y_train.values, label='True', color='salmon')
plt.plot(y_train_pred_lstm, label='LSTM Predicted', color='cornflowerblue')
plt.title('Training Set: True vs LSTM Predicted AQI')
plt.xlabel('Sample')
plt.ylabel('AQI')
plt.legend()
plt.show()

# 绘制测试集真实值与预测值对比图
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='True', color='salmon')
plt.plot(y_test_pred_lstm, label='LSTM Predicted', color='cornflowerblue')
plt.title('Test Set: True vs LSTM Predicted AQI')
plt.xlabel('Sample')
plt.ylabel('AQI')
plt.legend()
plt.show()


# # 设置全局字体大小
# plt.rcParams['font.size'] = 12

# # 绘制训练集真实值与预测值对比图
# plt.figure(figsize=(12, 6))
# plt.plot(y_train.values, label='True', color='blue', linewidth=2)
# plt.plot(y_train_pred_lstm, label='LSTM Predicted', color='orange', linestyle='--', linewidth=2)
# plt.title('Training Set: True vs LSTM Predicted AQI')
# plt.xlabel('Sample')
# plt.ylabel('AQI')
# plt.legend()
# plt.grid(True)
# plt.show()

# # 绘制测试集真实值与预测值对比图
# plt.figure(figsize=(12, 6))
# plt.plot(y_test.values, label='True', color='blue', linewidth=2)
# plt.plot(y_test_pred_lstm, label='LSTM Predicted', color='orange', linestyle='--', linewidth=2)
# plt.title('Test Set: True vs LSTM Predicted AQI')
# plt.xlabel('Sample')
# plt.ylabel('AQI')
# plt.legend()
# plt.grid(True)
# plt.show()
