import pandas as pd
from pgmpy.models import BayesianModel
from sklearn.model_selection import train_test_split
from pgmpy.estimators import MaximumLikelihoodEstimator

# 加载数据集
data = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/bayes_data.csv")

# 选择用于构建网络的特征
features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'bmi-level', 'age-level', 'ap-level']
data = data[features]

# 划分训练集和测试集，不删除缺失值
train_data, test_data = train_test_split(data, test_size=0.7, random_state=42)

# 删除训练集中的缺失值
train_data = train_data.dropna()

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
    ('gender', 'Physiological index'),  # 添加边连接 gender 到子类 Physiological index
    ('age-level', 'Physiological index'),  # 添加边连接 age-level 到子类 Physiological index
    ('cholesterol', 'Medical index'),  # 添加边连接 cholesterol 到子类 Medical index
    ('gluc', 'Medical index'),  # 添加边连接 gluc 到子类 Medical index
    ('ap-level', 'Medical index'),  # 添加边连接 ap-level 到子类 Medical index
    ('smoke', 'Subjective information'),  # 添加边连接 smoke 到子类 Subjective information
    ('alco', 'Subjective information'),  # 添加边连接 alco 到子类 Subjective information
    ('active', 'Subjective information'),  # 添加边连接 active 到子类 Subjective information
])

# 从训练集中估计参数
model.fit(train_data, estimator=MaximumLikelihoodEstimator)

# 输出节点及其 CPD
for node in model.nodes():
    cpd = model.get_cpds(node)
    print(f"CPD for Node {node}:\n{cpd}")

# 输出节点之间的结合关系
for edge in model.edges():
    print(f"Edge: {edge[0]} -> {edge[1]}")

# 输出节点的父节点
for node in model.nodes():
    parents = model.get_parents(node)
    print(f"Node: {node}, Parents: {parents}")
