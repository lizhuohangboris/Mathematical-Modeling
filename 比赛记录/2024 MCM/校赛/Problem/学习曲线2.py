import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集
data = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/bayes_data.csv")

# 选择特征
features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'bmi-level', 'age-level', 'ap-level']
data = data[features]

# 划分数据
X = data.drop('cardio', axis=1)
y = data['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# 创建贝叶斯网络模型
model = BayesianModel([
    ('gender', 'cardio'), 
    ('cholesterol', 'cardio'), 
    ('gluc', 'cardio'),
    ('smoke', 'cardio'), 
    ('alco', 'cardio'), 
    ('active', 'cardio'),
    ('bmi-level', 'cardio'), 
    ('age-level', 'cardio'), 
    ('ap-level', 'cardio'),
])

# 使用 MaximumLikelihoodEstimator 估算参数
model.fit(X_train, estimator=MaximumLikelihoodEstimator)

# 初始化列表以存储结果
train_scores = []
test_scores = []

# 定义不同的训练集大小
train_sizes = np.linspace(0.1, 1.0, 10)

# 循环不同的训练集大小
for size in train_sizes:
    # 从训练数据中取出子集
    subset_size = int(size * len(X_train))
    X_subset, _, y_subset, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)

    # 在子集上拟合模型
    model.fit(X_subset, estimator=MaximumLikelihoodEstimator)

    # 对训练集和测试集进行预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 计算准确度并追加到列表中
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_scores.append(train_accuracy)
    test_scores.append(test_accuracy)

# 绘制学习曲线
plt.plot(train_sizes, train_scores, 'o-', label='训练得分')
plt.plot(train_sizes, test_scores, 'o-', label='测试得分')
plt.xlabel('训练集大小')
plt.ylabel('准确度')
plt.title('自定义学习曲线')
plt.legend()
plt.show()
