import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel("D:/Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 统计建模/随机森林.xlsx")

# 将"Month"列转换为日期类型
data['Month'] = pd.to_datetime(data['Month'], format='%b-%y')

# 将"Month"列重命名为"ds"，将"AQI"列重命名为"y"，并创建一个Prophet模型
model_data = data.rename(columns={'Month': 'ds', 'AQI': 'y'})

# 创建Prophet模型
model = Prophet()

# 拟合模型
model.fit(model_data)

# 创建未来的日期
future = model.make_future_dataframe(periods=365)

# 进行预测
forecast = model.predict(future)

# 绘制预测结果
fig = model.plot(forecast)
plt.xlabel('Date')
plt.ylabel('AQI')
plt.title('Forecasted AQI')
plt.show()
