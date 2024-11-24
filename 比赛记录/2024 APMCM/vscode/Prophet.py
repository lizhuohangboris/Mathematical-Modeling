import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

# 忽略警告
warnings.simplefilter("ignore", category=FutureWarning)

# 读取数据
file_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 APMCM\数据\3.xlsx"
df = pd.read_excel(file_path)

# 清理列名
df.columns = df.columns.str.strip()  # 去除空格
df.columns = df.columns.str.replace('’', "'", regex=False)  # 替换全角引号为半角引号

# 准备数据
df_production = df[['Years', "Total Value of China's Pet Food Production"]].rename(columns={'Years': 'ds', "Total Value of China's Pet Food Production": 'y'})
df_exports = df[['Years', "Total Value of China's Pet Food Exports"]].rename(columns={'Years': 'ds', "Total Value of China's Pet Food Exports": 'y'})

# 将年份转为日期格式
df_production['ds'] = pd.to_datetime(df_production['ds'].astype(str) + '-01-01')
df_exports['ds'] = pd.to_datetime(df_exports['ds'].astype(str) + '-01-01')

# 创建 Prophet 模型并拟合数据
model_production = Prophet()
model_production.fit(df_production)

model_exports = Prophet()
model_exports.fit(df_exports)

# 生成未来 3 年的日期框架 (2024-2026)
future_production = model_production.make_future_dataframe(periods=3, freq='Y')  # 预测 3 年
future_exports = model_exports.make_future_dataframe(periods=3, freq='Y')  # 预测 3 年

# 进行预测
forecast_production = model_production.predict(future_production)
forecast_exports = model_exports.predict(future_exports)

# 打印预测结果
print("Production Forecast for 2024-2026:")
print(forecast_production[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3))
print("Exports Forecast for 2024-2026:")
print(forecast_exports[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3))

# 绘制预测结果
plt.figure(figsize=(14, 6))

# 生产预测图
plt.subplot(1, 2, 1)
plt.plot(df_production['ds'], df_production['y'], 'bo', label='Historical Data')  # 标注历史数据点
plt.plot(forecast_production['ds'], forecast_production['yhat'], 'r-', label='Forecast')  # 预测曲线
plt.fill_between(forecast_production['ds'], forecast_production['yhat_lower'], forecast_production['yhat_upper'], color='pink', alpha=0.3, label='Uncertainty Interval')  # 置信区间
plt.title("Pet Food Production Value Forecast (2024-2026)")
plt.xlabel("Year")
plt.ylabel("Production Value (CNY)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # 设置年份格式
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))  # 每年显示一次
plt.legend()

# 在历史数据点上添加数值标注
for i in range(len(df_production)):
    plt.text(df_production['ds'].iloc[i], df_production['y'].iloc[i], f"{df_production['y'].iloc[i]:.2f}", fontsize=9, ha='right', color='blue')

# 在预测数据点上添加数值标注
for i in range(len(forecast_production)):
    if forecast_production['ds'].iloc[i] > df_production['ds'].max():  # 只标注未来的预测值
        plt.text(forecast_production['ds'].iloc[i], forecast_production['yhat'].iloc[i], f"{forecast_production['yhat'].iloc[i]:.2f}", fontsize=9, ha='right', color='red')

# 出口预测图
plt.subplot(1, 2, 2)
plt.plot(df_exports['ds'], df_exports['y'], 'bo', label='Historical Data')  # 标注历史数据点
plt.plot(forecast_exports['ds'], forecast_exports['yhat'], 'r-', label='Forecast')  # 预测曲线
plt.fill_between(forecast_exports['ds'], forecast_exports['yhat_lower'], forecast_exports['yhat_upper'], color='pink', alpha=0.3, label='Uncertainty Interval')  # 置信区间
plt.title("Pet Food Export Value Forecast (2024-2026)")
plt.xlabel("Year")
plt.ylabel("Export Value (USD)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # 设置年份格式
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))  # 每年显示一次
plt.legend()

# 在历史数据点上添加数值标注
for i in range(len(df_exports)):
    plt.text(df_exports['ds'].iloc[i], df_exports['y'].iloc[i], f"{df_exports['y'].iloc[i]:.2f}", fontsize=9, ha='right', color='blue')

# 在预测数据点上添加数值标注
for i in range(len(forecast_exports)):
    if forecast_exports['ds'].iloc[i] > df_exports['ds'].max():  # 只标注未来的预测值
        plt.text(forecast_exports['ds'].iloc[i], forecast_exports['yhat'].iloc[i], f"{forecast_exports['yhat'].iloc[i]:.2f}", fontsize=9, ha='right', color='red')

plt.tight_layout()
plt.show()
