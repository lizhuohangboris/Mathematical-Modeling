import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# 加载数据集
data = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/output.csv")  # 替换为实际文件路径

# 数据划分
X = data[['bmi-level', 'cholesterol']]  # 选择两个特征，你可以根据需要调整
y = data['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型训练
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 输出模型相关特征值
print("模型系数（Coefficients）:", model.coef_)
print("模型截距（Intercept）:", model.intercept_)

# 模型预测
y_pred = model.predict(X_test_scaled)

# 模型评估
print("\n模型评估（Model Evaluation）:")
print("准确率（Accuracy）:", accuracy_score(y_test, y_pred))
print("混淆矩阵（Confusion Matrix）:\n", confusion_matrix(y_test, y_pred))
print("分类报告（Classification Report）:\n", classification_report(y_test, y_pred))

# 绘制逻辑回归的决策边界
def plot_decision_boundary(model, X, y):
    h = .02  # 步长
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k', marker='o', s=30, linewidth=0.8)

    plt.title('Decision Boundary of Logistic Regression')
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.show()

# 使用两个特征进行逻辑回归决策边界可视化
plot_decision_boundary(model, X_train, y_train)
