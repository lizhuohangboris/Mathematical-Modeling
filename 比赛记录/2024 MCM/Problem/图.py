import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# 加载数据集
data = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/cardio_train.csv")

# 选择用于预测的特征
features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

# 对类别特征进行独热编码
data_encoded = pd.get_dummies(data[features])

# 将编码后的数据和目标变量整合
X = pd.concat([data_encoded, data.drop(features, axis=1)], axis=1)
y = data['cardio']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用高斯朴素贝叶斯进行训练
nb_model = GaussianNB()

# 绘制学习曲线
train_sizes, train_scores, test_scores = learning_curve(nb_model, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

# 计算平均值和标准差
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Score', marker='o')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2)

plt.plot(train_sizes, test_scores_mean, label='Cross-Validation Score', marker='x')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2)

# 添加标题和标签
plt.title('Learning Curve - Training and Cross-Validation Scores')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')

# 显示图例
plt.legend()

# 显示图表
plt.show()
