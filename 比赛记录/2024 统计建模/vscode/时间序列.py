import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel("D:/Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 统计建模/随机森林.xlsx")

# 将日期列转换为日期时间类型并设置为索引
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# 准备目标变量
y = data["AQI"]

# 拆分训练集和测试集
train_size = int(len(y) * 0.8)
train, test = y[:train_size], y[train_size:]

# 训练ARIMA模型
model = ARIMA(train, order=(1, 1, 1))  # 使用ARIMA(1,1,1)模型，可以根据需要调整参数
model_fit = model.fit()

# 预测
predictions = model_fit.forecast(steps=len(test))[0]

# 计算均方误差
mse = ((test - predictions) ** 2).mean()
print("Mean Squared Error:", mse)

# 绘制预测结果
plt.plot(train.index, train, label='Train', color='blue')
plt.plot(test.index, test, label='Test', color='green')
plt.plot(test.index, predictions, label='Predictions', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.title('Actual vs. Predicted AQI (ARIMA)')
plt.legend()
plt.show()
