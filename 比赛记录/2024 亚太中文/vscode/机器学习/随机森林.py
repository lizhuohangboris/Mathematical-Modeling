import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

# 确认文件路径正确无误
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\第二问.csv'

# 加载数据，尝试使用逗号作为分隔符
data = pd.read_csv(file_path, delimiter=',', encoding='gbk')

# 查看列名，去除可能存在的空格
data.columns = data.columns.str.strip()

# 分离特征和目标变量
X = data.drop('洪水概率', axis=1)
y = data['洪水概率']

# 将目标变量转换为数值型（回归任务）
y = y.astype('float')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, max_features=4, min_samples_split=3,
                              min_samples_leaf=1, criterion='squared_error', max_depth=2, random_state=42)
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出随机森林模型的参数
print(f'Number of Decision Trees: {model.n_estimators}')
print(f'Max Features per Tree: {model.max_features}')
print(f'Minimum Samples to Split a Node: {model.min_samples_split}')
print(f'Minimum Samples per Leaf Node: {model.min_samples_leaf}')
print(f'Criterion: {model.criterion}')
print(f'Maximum Depth of Trees: {model.max_depth}')

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 输出第一棵树的结构
estimator = model.estimators_[0]

# 导出为dot文件
dot_data = export_graphviz(estimator, out_file=None,
                           feature_names=X.columns,
                           filled=True, rounded=True,
                           special_characters=True)

# 使用graphviz渲染决策树图形
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# 保存决策树图形为文件
with open("tree.png", "wb") as f:
    f.write(graph.create_png())
