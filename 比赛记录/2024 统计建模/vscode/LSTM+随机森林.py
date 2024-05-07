import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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

# LSTM 模型
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将数据重塑为 LSTM 所需的形状 (samples, time steps, features)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 训练 LSTM 模型
model.fit(X_train_reshaped, y_train, epochs=100, verbose=0)

# 使用 LSTM 模型进行预测
y_train_pred_lstm = model.predict(X_train_reshaped)
y_test_pred_lstm = model.predict(X_test_reshaped)

# 将 LSTM 的预测结果作为特征输入随机森林模型
X_train_with_lstm = np.hstack((X_train_scaled, y_train_pred_lstm))
X_test_with_lstm = np.hstack((X_test_scaled, y_test_pred_lstm))

# 随机森林模型
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 拟合模型
model_rf.fit(X_train_with_lstm, y_train)

# 预测训练集和测试集的AQI值
y_train_pred_rf = model_rf.predict(X_train_with_lstm)
y_test_pred_rf = model_rf.predict(X_test_with_lstm)

# 计算模型在训练集和测试集上的R^2分数
train_score_rf = model_rf.score(X_train_with_lstm, y_train)
test_score_rf = model_rf.score(X_test_with_lstm, y_test)
print("Random Forest Train R^2 Score:", train_score_rf)
print("Random Forest Test R^2 Score:", test_score_rf)


