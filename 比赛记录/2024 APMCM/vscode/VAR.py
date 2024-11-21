import pandas as pd
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置 Matplotlib 显示中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 数据集
data = {
    "Years": [2019, 2020, 2021, 2022, 2023],
    "Total Value of China’s Pet Food Production": [440.7, 727.3, 1554, 1508, 2793],
    "Total Value of China’s Pet Food Exports": [154.1, 70.952, 88.328, 178.828, 286.704]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 将年份设置为索引
df.set_index("Years", inplace=True)

# 选择模型变量
variables = df[["Total Value of China’s Pet Food Production", "Total Value of China’s Pet Food Exports"]]

# 拆分训练集
train = variables

# 建立 VAR 模型
model = VAR(train)
results = model.fit(maxlags=2)  # 自动选择滞后阶数，或手动设置 maxlags=2

# 输出模型摘要
print(results.summary())

# 预测未来三年
forecast_steps = 3
forecast = results.forecast(y=train.values, steps=forecast_steps)

# 创建未来三年的年份列表
future_years = [2024, 2025, 2026]
forecast_df = pd.DataFrame(forecast, columns=train.columns, index=future_years)

# 打印预测结果
print("\n未来三年的预测值：")
print(forecast_df)

# 可视化
plt.figure(figsize=(10, 6))

# 生产总值
plt.plot(df.index, df["Total Value of China’s Pet Food Production"], label="历史数据: 生产总值", marker="o")
plt.plot(future_years, forecast_df["Total Value of China’s Pet Food Production"], label="预测数据: 生产总值", color="red", marker="o")
for year, value in zip(future_years, forecast_df["Total Value of China’s Pet Food Production"]):
    plt.text(year, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)

# 出口总值
plt.plot(df.index, df["Total Value of China’s Pet Food Exports"], label="历史数据: 出口总值", marker="o")
plt.plot(future_years, forecast_df["Total Value of China’s Pet Food Exports"], label="预测数据: 出口总值", color="blue", marker="o")
for year, value in zip(future_years, forecast_df["Total Value of China’s Pet Food Exports"]):
    plt.text(year, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)

# 图例和标题
plt.legend()
plt.title("中国宠物食品生产总值与出口总值预测")
plt.xlabel("年份")
plt.ylabel("总值（亿元）")
plt.grid()
plt.show()
