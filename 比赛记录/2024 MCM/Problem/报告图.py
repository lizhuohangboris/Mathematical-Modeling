import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

# 模拟数据
precision = [0.71, 0.75]
recall = [0.79, 0.66]
f1_score = [0.75, 0.71]
support = [24328, 23778]
class_labels = ['Class 0', 'Class 1']

# 创建一个条形图
fig, ax = plt.subplots(figsize=(8, 6))

bar_width = 0.2
index = np.arange(len(class_labels))

# 绘制条形图
bar1 = ax.bar(index, precision, bar_width, label='Precision')
bar2 = ax.bar(index + bar_width, recall, bar_width, label='Recall')
bar3 = ax.bar(index + 2*bar_width, f1_score, bar_width, label='F1-Score')

# 添加支持数
for i, v in enumerate(support):
    ax.text(i + 0.5*bar_width, v + 100, str(v), color='black', ha='center')

# 设置图表标题和标签
ax.set_title('Classification Report')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(class_labels)
ax.legend()

# 显示图形
plt.show()
