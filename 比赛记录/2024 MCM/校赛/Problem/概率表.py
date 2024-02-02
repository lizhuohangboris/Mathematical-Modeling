import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

# 加载数据集
data = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/cardio_train.csv")

# 选择用于构建网络的特征
features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
data = data[features]

# 创建贝叶斯网络模型
model = BayesianModel([('gender', 'cardio'), ('cholesterol', 'cardio'), ('gluc', 'cardio'),
                       ('smoke', 'cardio'), ('alco', 'cardio'), ('active', 'cardio')])

# 从数据中估计参数
model.fit(data, estimator=MaximumLikelihoodEstimator)

# 输出节点的概率表
for cpd in model.get_cpds():
    print(f"\nCPD for {cpd.variable} given {', '.join(cpd.variables[1:])}:\n")
    print(cpd)
