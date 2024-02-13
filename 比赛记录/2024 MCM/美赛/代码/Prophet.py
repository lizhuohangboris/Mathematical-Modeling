from prophet import Prophet
import pandas as pd

# 读取数据集
df = pd.read_csv('C:/Users/92579/Desktop/2024_MCM-ICM_Problems/2024_MCM-ICM_Problems/Wimbledon_featured_matches.csv')

# 创建并拟合Prophet模型
model = Prophet()
model.fit(df)

# 创建一个包含未来日期的DataFrame，用于预测
future = model.make_future_dataframe(periods=365)  # 365天的预测，你可以根据需要调整

# 进行预测
forecast = model.predict(future)

# 查看预测结果
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
