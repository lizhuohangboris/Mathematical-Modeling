import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly

# 读取CSV文件
file_path = 'C:/Users/92579/Documents/GitHub/Mathematical-Modeling/学习记录/2022美赛/C题/2022_Problem_C_DATA/BCHAIN-MKPRU.csv'
df = pd.read_csv(file_path)

# 转换日期格式
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')  # 使用适当的日期格式
df.rename(columns={'Date': 'ds', 'Value': 'y'}, inplace=True)

# 创建并拟合Prophet模型
m = Prophet()
m.fit(df)

# 生成未来时间点
future = m.make_future_dataframe(periods=365)

# 进行预测
forecast = m.predict(future)

# 绘制预测结果图表
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

# 使用Plotly进行交互式可视化
fig3 = plot_plotly(m, forecast)
fig4 = plot_components_plotly(m, forecast)

# 实际值
y_true = df['y'].values

# 预测值
y_pred = forecast['yhat'].values[-len(y_true):]

# 计算均方误差（MSE）
mse = mean_squared_error(y_true, y_pred)

# 计算均方根误差（RMSE）
rmse = np.sqrt(mse)

# 计算平均绝对误差（MAE）
mae = mean_absolute_error(y_true, y_pred)

# 计算平均绝对百分比误差（MAPE）
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 打印评估指标
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

# 直接显示图表
fig1.show()
input("Press Enter to continue...")

fig2.show()
input("Press Enter to continue...")

fig3.show()
input("Press Enter to continue...")

fig4.show()
input("Press Enter to continue...")
