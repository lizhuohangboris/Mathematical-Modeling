import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 模型名称
models = ['Logistic Regression', 'Bayesian Model', 'XGBoost']

# 准确率数据
accuracy_values = [ 0.7236261337730999, 0.7930434782608695, 0.7283794926516952]

# 混淆矩阵数据
conf_matrices = [
    np.array([[8149, 2239], [3459, 6770]]),
    np.array([[9285, 1213], [3071, 7131]]),
    np.array([[7971, 2417], [3183, 7046]])
]

# 准确率可视化
plt.figure(figsize=(10, 5))
plt.bar(models, accuracy_values, color=['royalblue', 'limegreen', 'khaki'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# 混淆矩阵可视化
plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    sns.heatmap(conf_matrices[i], annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.title(f'Confusion Matrix - {models[i]}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 模型名称
models = ['Logistic Regression', 'Bayesian Model', 'XGBoost']

# 准确率数据
accuracy_values = [ 0.7236261337730999, 0.7930434782608695, 0.7283794926516952]

# 准确率可视化
plt.figure(figsize=(10, 5))
bars = plt.bar(models, accuracy_values, color=['royalblue', 'limegreen', 'khaki'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# 在每个条形的顶部标上具体的数值
for bar, accuracy in zip(bars, accuracy_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{accuracy:.3f}', 
             ha='center', va='bottom', color='black', fontsize=10)

plt.show()
