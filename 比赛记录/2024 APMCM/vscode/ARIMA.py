import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置 Matplotlib 显示中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 读取 Excel 数据
file_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 APMCM\数据\2.xlsx"
data = pd.read_excel(file_path)

# 确保年份列是时间索引
data.set_index('Years', inplace=True)

# 提取目标变量：全球宠物食品市场规模
y = data['全球宠物食品市场规模（十亿美元）']

# 创建年份列表
years = y.index.tolist()

# 拆分训练集
train = y[:len(y)-1]

# 建立 ARIMA 模型
model = ARIMA(train, order=(1, 1, 1))  # 初始参数 (p=1, d=1, q=1)
model_fit = model.fit()

# 输出模型摘要
print(model_fit.summary())

# 预测未来三年的值
forecast_steps = 3
forecast = model_fit.forecast(steps=forecast_steps)

# 创建未来三年的年份列表
future_years = [years[-1] + i for i in range(1, forecast_steps + 1)]

# 打印预测值
print("未来三年的预测值：")
for year, value in zip(future_years, forecast):
    print(f"{year}: {value:.2f}")

# 可视化
plt.figure(figsize=(10, 6))

# 历史数据
plt.plot(years, y, label='历史数据', marker='o')
for i, (year, value) in enumerate(zip(years, y)):
    plt.text(year, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)

# 预测数据
plt.plot(future_years, forecast, label='预测数据', color='red', marker='o')
for i, (year, value) in enumerate(zip(future_years, forecast)):
    plt.text(year, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)

# 图例和标题
plt.legend()
plt.title('全球宠物食品市场规模预测')
plt.xlabel('年份')
plt.ylabel('市场规模（十亿美元）')
plt.grid()
plt.show()
