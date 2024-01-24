import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tabulate import tabulate  # 导入 tabulate 模块

# 加载数据集
data = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/output.csv")  # 替换为实际文件路径

# 数据划分
X = data.drop('cardio', axis=1)
y = data['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型训练
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 模型预测
y_pred = model.predict(X_test_scaled)

# 模型评估
print("\n模型评估（Model Evaluation）:")
print("准确率（Accuracy）:", accuracy_score(y_test, y_pred))
print("混淆矩阵（Confusion Matrix）:\n", confusion_matrix(y_test, y_pred))

# 获取Classification Report字符串
class_report_str = classification_report(y_test, y_pred)

# 加入表格线并输出
print("Classification Report:\n")
print(tabulate([line.split() for line in class_report_str.split('\n')], headers='keys', tablefmt='pretty'))

