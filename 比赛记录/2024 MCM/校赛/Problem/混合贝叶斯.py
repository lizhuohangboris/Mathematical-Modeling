import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.estimators import KernelDensity
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# 加载数据集
data = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/bayes_data.csv")

# 选择用于构建网络的特征
features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'bmi-level', 'age-level', 'ap-level', 'ap_hi', 'ap_lo', 'height', 'weight']
data = data[features]

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.7, random_state=42)

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
    ('ap_hi', 'ap-level'),  # 新增边缘连接
    ('ap_lo', 'ap-level'),  # 新增边缘连接
    ('height', 'bmi-level'),  # 新增边缘连接
    ('weight', 'bmi-level'),  # 新增边缘连接
])

# 从训练集中估计参数
for node in model.nodes():
    if node not in ['ap_hi', 'ap_lo', 'height', 'weight']:
        model.fit(train_data, estimator=MaximumLikelihoodEstimator, state_names={node: list(train_data[node].unique())})
    else:
        # 使用 statsmodels 库进行 KDE
        kde = sm.nonparametric.KDEMultivariate(data=train_data[node], var_type='c', bw='normal_reference')
        pdf_values = kde.pdf(test_data[node].values)  # 计算在测试集上的概率密度值
        pdf_values /= pdf_values.sum()  # 将概率密度值归一化，确保和为1

        # 创建 ContinuousFactor 对象并添加 CPD
        continuous_factor = ParameterEstimator(KernelDensity(bandwidth=kde.bw, kernel='gau')).estimate(node, train_data)
        continuous_factor.set_values(pdf_values)
        model.add_cpds(continuous_factor)

# 输出节点之间的结合关系
for edge in model.edges():
    print(f"Edge: {edge[0]} -> {edge[1]}")

# 输出节点的父节点
for node in model.nodes():
    parents = model.get_parents(node)
    print(f"Node: {node}, Parents: {parents}")
