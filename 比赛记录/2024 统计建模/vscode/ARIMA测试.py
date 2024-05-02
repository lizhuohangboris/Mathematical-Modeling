import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# 从Excel文件中读取数据
data = pd.read_excel('D:/Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 统计建模/天津空气质量指数月统计历史数据2013.12-2024.5.xlsx')

# 数据预处理，将日期转换成datetime格式，设置为索引
data['Month'] = pd.to_datetime(data['Month'], format='%b-%y')
data.set_index('Month', inplace=True)

# 拆分数据为训练集和测试集
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# ARIMA模型训练和预测
history = [x for x in train['AQI']]
predictions = []
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))  # 选择ARIMA模型的参数（p,d,q）
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test['AQI'].iloc[t]
    history.append(obs)
    print('预测=%f, 实际=%f' % (yhat, obs))

# 评估预测准确性
error = mean_squared_error(test['AQI'], predictions)
print('测试集均方误差: %.3f' % error)
