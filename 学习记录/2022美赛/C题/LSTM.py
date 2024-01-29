# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 读取CSV文件
file_path = "C:/Users/92579/Documents/GitHub/Mathematical-Modeling/学习记录/2022美赛/C题/2022_Problem_C_DATA/BCHAIN-MKPRU.csv"
df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# 数据预处理
df = df.dropna()  # 删除缺失值
df = df.resample('D').mean()  # 将数据按天重采样，取均值

# 归一化数据
scaler = MinMaxScaler()
df['Value'] = scaler.fit_transform(df[['Value']])

# 划分训练集和测试集
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# 创建窗口数据集
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 10  # 可调整窗口大小
trainX, trainY = create_dataset(np.array(train), look_back)
testX, testY = create_dataset(np.array(test), look_back)

# 调整输入数据的形状
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)

# 在训练集和测试集上进行预测
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# 反归一化预测值
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# 计算训练集和测试集的均方根误差
trainScore = np.sqrt(np.mean(np.square(trainY[0] - trainPredict[:, 0])))
testScore = np.sqrt(np.mean(np.square(testY[0] - testPredict[:, 0])))

print('Train Score: %.2f RMSE' % (trainScore))
print('Test Score: %.2f RMSE' % (testScore))

# 可视化训练集的预测结果
trainPredictPlot = np.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

# 可视化测试集的预测结果
testPredictPlot = np.empty_like(df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2):len(df), :] = testPredict

# 将原始数据、训练集和测试集的预测结果进行可视化
plt.figure(figsize=(10, 6))
plt.plot(scaler.inverse_transform(df[['Value']]), label='Original Data')
plt.plot(trainPredictPlot, label='Train Predictions')
plt.plot(testPredictPlot, label='Test Predictions')
plt.title('Bitcoin Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
