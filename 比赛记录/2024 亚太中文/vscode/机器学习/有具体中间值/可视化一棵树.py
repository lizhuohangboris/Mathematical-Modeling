import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree, export_graphviz

# 读取数据集，指定编码格式
data = pd.read_csv(r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\第二问.csv', encoding='gbk')

# 查看数据集结构
print(data.head())

# 假设最后一列为标签，其余列为特征
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 训练随机森林回归模型
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X, y)

# 可视化随机森林中的一棵决策树
plt.figure(figsize=(20,10))
plot_tree(random_forest.estimators_[0], filled=True)
plt.show()

# 导出决策树为Graphviz格式
export_graphviz(random_forest.estimators_[0], out_file='tree.dot', 
                filled=True, rounded=True, 
                feature_names=data.columns[:-1], 
                class_names=None)

# 使用Graphviz命令将tree.dot文件转换为图像格式，例如：
# dot -Tpng tree.dot -o tree.png
