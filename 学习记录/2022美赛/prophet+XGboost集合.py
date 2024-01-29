import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor, DMatrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 加载数据，替换成你的实际数据路径
df = pd.read_csv('C:/Users/92579/Documents/GitHub/Mathematical-Modeling/学习记录/2022美赛/C题/2022_Problem_C_DATA/BCHAIN-MKPRU.csv')

# 将日期字符串转换为日期时间格式
df['Date'] = pd.to_datetime(df['Date'])

# 创建 Prophet 模型
m_prophet = Prophet()

# 加载数据
df_prophet = df.rename(columns={'Date': 'ds', 'Value': 'y'})
m_prophet.fit(df_prophet)

# 预测未来时间
future_prophet = m_prophet.make_future_dataframe(periods=365)
forecast_prophet = m_prophet.predict(future_prophet)

# 提取 Prophet 预测的趋势部分
trend_prophet = forecast_prophet['trend'].values[:len(df_prophet)]

# 计算残差
residual = df_prophet['y'].values - trend_prophet

# 创建 XGBoost 模型
xgb_model = XGBRegressor()

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(df_prophet['ds'].values, residual, test_size=0.2, random_state=42)

# 将数据转换为 DMatrix 格式
dtrain = DMatrix(np.array(X_train).reshape(-1, 1), label=y_train)
dtest = DMatrix(np.array(X_test).reshape(-1, 1), label=y_test)

# 训练 XGBoost 模型
xgb_model.fit(dtrain)

# 预测残差
residual_pred = xgb_model.predict(dtest)

# 计算 XGBoost 模型的均方根误差
rmse_xgb = np.sqrt(mean_squared_error(y_test, residual_pred))
print(f"XGBoost 模型的均方根误差: {rmse_xgb}")

# 集成模型预测
final_prediction = trend_prophet + xgb_model.predict(DMatrix(np.array(future_prophet['ds']).reshape(-1, 1)))

# 将结果添加到数据框中
df['Prophet_XGBoost_Prediction'] = np.concatenate([np.zeros(len(df)-len(final_prediction)), final_prediction])

# 打印最后几行数据，查看预测结果
print(df.tail())
