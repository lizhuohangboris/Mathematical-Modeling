import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 模型名称
models = ['Logistic Regression', 'Bayesian Model', 'XGBoost']

# 准确率数据
accuracy = [0.7269, 0.7946, 0.7265]

# 混淆矩阵数据
confusion_matrices = [
    [[8149, 2239], [3459, 6770]],
    [[6226, 812], [2022, 4740]],
    [[5339, 1582], [2177, 4647]]
]

# 准确率可视化
plt.figure(figsize=(10, 5))
plt.bar(models, accuracy, color=['blue', 'green', 'orange'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# 混淆矩阵可视化
plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    sns.heatmap(confusion_matrices[i], annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.title(f'Confusion Matrix - {models[i]}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
plt.tight_layout()
plt.show()
