# 导入必要的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 加载数据集
data = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/cardio_train.csv")
print(data.columns)

# 选择特征和目标变量
X = data.drop(['cardio'], axis=1)
y = data['cardio']

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建决策树模型
tree_model = DecisionTreeClassifier(random_state=42)

# 训练模型
tree_model.fit(X_train_scaled, y_train)

# 预测
y_pred_tree = tree_model.predict(X_test_scaled)

# 评估模型
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"决策树准确度: {accuracy_tree}")

# 打印分类报告
print("分类报告:")
print(classification_report(y_test, y_pred_tree))
