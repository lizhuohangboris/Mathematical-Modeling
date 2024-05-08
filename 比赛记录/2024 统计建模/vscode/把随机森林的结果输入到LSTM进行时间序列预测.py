import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

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

# 随机森林模型
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# 使用随机森林模型对训练集和测试集进行预测
y_train_pred_rf = model_rf.predict(X_train)
y_test_pred_rf = model_rf.predict(X_test)

# 将随机森林模型的预测结果作为特征加入原始数据集
X_train["RF_Prediction"] = y_train_pred_rf
X_test["RF_Prediction"] = y_test_pred_rf

# LSTM 模型
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将数据重塑为 LSTM 所需的形状 (samples, time steps, features)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')

# 训练 LSTM 模型
model_lstm.fit(X_train_reshaped, y_train, epochs=100, verbose=0)

# 使用 LSTM 模型进行预测
y_train_pred_lstm = model_lstm.predict(X_train_reshaped)
y_test_pred_lstm = model_lstm.predict(X_test_reshaped)

# 计算模型在训练集和测试集上的R²分数
train_score_lstm = model_lstm.evaluate(X_train_reshaped, y_train, verbose=0)
test_score_lstm = model_lstm.evaluate(X_test_reshaped, y_test, verbose=0)
print("LSTM Train R² Score:", train_score_lstm)
print("LSTM Test R² Score:", test_score_lstm)




# 绘制训练集上的趋势图
plt.figure(figsize=(10, 6))
plt.plot(X_train.index, y_train, label='实际AQI值', color='blue')
plt.plot(X_train.index, y_train_pred_lstm, label='LSTM预测AQI值', color='red')
plt.xlabel('日期')
plt.ylabel('AQI值')
plt.title('训练集上的AQI值趋势图')
plt.legend()
plt.show()

# 绘制测试集上的趋势图
plt.figure(figsize=(10, 6))
plt.plot(X_test.index, y_test, label='实际AQI值', color='blue')
plt.plot(X_test.index, y_test_pred_lstm, label='LSTM预测AQI值', color='red')
plt.xlabel('日期')
plt.ylabel('AQI值')
plt.title('测试集上的AQI值趋势图')
plt.legend()
plt.show()
