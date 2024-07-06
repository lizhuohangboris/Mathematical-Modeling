import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import export_text
import numpy as np

# 确认文件路径正确无误
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\第二问.csv'

# 加载数据，尝试使用逗号作为分隔符
data = pd.read_csv(file_path, delimiter=',', encoding='gbk')

# 查看列名，去除可能存在的空格
data.columns = data.columns.str.strip()

# 分离特征和目标变量
X = data.drop('洪水概率', axis=1)
y = data['洪水概率']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# 输出特征重要性
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(importance_df)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# 输出单棵树的结构
# 获取模型中的第一棵树
tree = model.estimators_[0]
tree_structure = export_text(tree, feature_names=list(features))
print("Tree Structure (First Tree):")
print(tree_structure)

# 输出决策路径
sample_id = 0
sample = X_test.iloc[sample_id].values.reshape(1, -1)
decision_path = model.decision_path(sample)
print(f"Decision Path for Sample {sample_id}:")
print(decision_path)

# 输出叶节点的预测分布
leaf_nodes = model.apply(X_test)
unique_leaf_nodes, leaf_counts = np.unique(leaf_nodes, return_counts=True)
leaf_predictions = {leaf: [] for leaf in unique_leaf_nodes}

for i, leaf in enumerate(leaf_nodes):
    leaf_predictions[leaf].append(y_test.iloc[i])

print("Leaf Node Prediction Distribution:")
for leaf, predictions in leaf_predictions.items():
    print(f"Leaf Node {leaf}: Predictions: {predictions}, Count: {leaf_counts[unique_leaf_nodes == leaf][0]}")
