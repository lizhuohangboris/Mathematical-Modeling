import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 数据
years = [2019, 2020, 2021, 2022, 2023]
cat_population = [4412, 4862, 5806, 6536, 6980]
dog_population = [5503, 5222, 5429, 5119, 5175]
market_size = [2191, 2259, 2733, 3069, 3264]  # 单位：亿元
food_market_size = [116.4, 138.2, 155.4, 173.2, 190]  # 单位：亿元

# 绘制图表
plt.figure(figsize=(12, 6))

# 绘制猫和狗数量变化曲线
plt.plot(years, cat_population, marker='o', label='猫数量（万只）', linewidth=2)
plt.plot(years, dog_population, marker='o', label='狗数量（万只）', linewidth=2)

# 绘制市场规模柱状图
bars_market = plt.bar([x - 0.2 for x in years], market_size, alpha=0.6, label='宠物行业市场规模（亿元）', color='gray', width=0.4)
bars_food = plt.bar([x + 0.2 for x in years], food_market_size, alpha=0.6, label='宠物食物市场规模（亿元）', color='green', width=0.4)

# 在柱状图上添加数据标签
for bar in bars_market:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 50, f'{int(height)}', ha='center', fontsize=10)

for bar in bars_food:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 5, f'{height:.1f}', ha='center', fontsize=10)

# 在猫和狗曲线数据点上添加标签
for x, y in zip(years, cat_population):
    plt.text(x, y + 100, f'{y}', ha='center', fontsize=10, color='blue')
for x, y in zip(years, dog_population):
    plt.text(x, y + 100, f'{y}', ha='center', fontsize=10, color='orange')

# 添加图表标题和轴标签
plt.title('2019-2023年中国宠物行业及宠物食物市场发展趋势', fontsize=16)
plt.xlabel('年份', fontsize=12)
plt.ylabel('数量（万只）/ 市场规模（亿元）', fontsize=12)
plt.xticks(years)
plt.legend(fontsize=12)
plt.grid(alpha=0.5, linestyle='--')

# 显示图表
plt.tight_layout()
plt.show()
