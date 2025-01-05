import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
# 原始预测结果（即未增加或减少植物）
original_predictions = [42.34, 100.77, 77.16, 18.42, 38.45, 15.08, 30.46, 32.77, 17.76, 6.72]

# 模拟植物增加后的预测结果
increased_plant_predictions = [38.53, 27.74, 40.4, 17.25, 27.61, 35.51, 18.62, 33.51, 48.39, 98.14]

# 模拟植物减少后的预测结果
decreased_plant_predictions = [38.53, 27.74, 40.4, 17.25, 28.85, 32.42, 18.6, 34.53, 47.44, 97.94]

# 1. 柱状图：比较三种情况（原始、植物增加、植物减少）下的鸟类种群数量
labels = [f"Point {i+1}" for i in range(len(original_predictions))]
x = np.arange(len(labels))

# 设置柱状图宽度
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制三组数据的柱状图
rects1 = ax.bar(x - width, original_predictions, width, label='Original', color='lightblue')
rects2 = ax.bar(x, increased_plant_predictions, width, label='Increased Plants', color='lightgreen')
rects3 = ax.bar(x + width, decreased_plant_predictions, width, label='Decreased Plants', color='salmon')

# 设置标签和标题
ax.set_xlabel('Region Points', fontsize=12)
ax.set_ylabel('Bird Population Predictions', fontsize=12)
ax.set_title('Effect of Plant Change on Bird Population', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

# 显示图表
plt.tight_layout()
plt.show()

# 2. 箱线图：展示三种情境下的鸟类种群数量分布
data = [original_predictions, increased_plant_predictions, decreased_plant_predictions]

fig, ax = plt.subplots(figsize=(8, 6))

# 绘制箱线图
ax.boxplot(data, labels=['Original', 'Increased Plants', 'Decreased Plants'], patch_artist=True, 
           boxprops=dict(facecolor='lightblue', color='blue'), 
           whiskerprops=dict(color='blue'), capprops=dict(color='blue'))

# 设置标题和标签
ax.set_title('Distribution of Bird Population Predictions for Different Plant Conditions', fontsize=14)
ax.set_ylabel('Bird Population Predictions', fontsize=12)

# 显示图表
plt.tight_layout()
plt.show()

# 3. 散点图：展示每个点在不同植物变化情境下的预测值
plt.figure(figsize=(8, 6))

# 绘制散点图
plt.scatter(range(1, len(original_predictions) + 1), original_predictions, label='Original', color='lightblue', marker='o')
plt.scatter(range(1, len(increased_plant_predictions) + 1), increased_plant_predictions, label='Increased Plants', color='lightgreen', marker='s')
plt.scatter(range(1, len(decreased_plant_predictions) + 1), decreased_plant_predictions, label='Decreased Plants', color='salmon', marker='^')

# 设置标签和标题
plt.xlabel('Region Points', fontsize=12)
plt.ylabel('Bird Population Predictions', fontsize=12)
plt.title('Effect of Plant Change on Bird Population (Scatter Plot)', fontsize=14)
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()
