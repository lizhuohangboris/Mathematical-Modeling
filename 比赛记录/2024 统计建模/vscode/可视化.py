import matplotlib.pyplot as plt

# 训练和测试R^2得分
methods = ['Random Forest', 'XGBoost', 'LightGBM']
train_scores = [0.9757477919112507, 0.9999999971717937, 0.8766610200363871]
test_scores = [0.9208221928727248, 0.8999337510900779, 0.7713688256175901]

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制柱状图
bar_width = 0.35
index = range(len(methods))
bars1 = plt.bar(index, train_scores, bar_width, color='bisque', label='Train R^2 Score')
bars2 = plt.bar([i + bar_width for i in index], test_scores, bar_width, color='salmon', label='Test R^2 Score')

# 添加数字标签
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

# 添加标签和标题
plt.xlabel('Method')
plt.ylabel('R^2 Score')
plt.title('Comparison of Methods')
plt.xticks([i + bar_width/2 for i in index], methods)
plt.legend()

# 显示图形
plt.grid(True)
plt.tight_layout()
plt.show()
