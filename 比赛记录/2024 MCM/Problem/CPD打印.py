import pandas as pd
from pgmpy.models import BayesianModel
from sklearn.model_selection import train_test_split  # Add this line
from pgmpy.estimators import MaximumLikelihoodEstimator

# 加载数据集
data = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/bayes_data.csv")

# 选择用于构建网络的特征
features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'bmi-level', 'age-level', 'ap-level']
data = data[features]

# 划分训练集和测试集
train_data, test_data = train_test_split(data.dropna(), test_size=0.7, random_state=42)

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
    ('gender', 'Physiological index'),
    ('age-level', 'Physiological index'),
    ('cholesterol', 'Medical index'),
    ('gluc', 'Medical index'),
    ('ap-level', 'Medical index'),
    ('smoke', 'Subjective information'),
    ('alco', 'Subjective information'),
    ('active', 'Subjective information'),
])

# 从训练集中估计参数
model.fit(train_data, estimator=MaximumLikelihoodEstimator)

# 创建一个字典来存储 CPD 数据
cpd_data = {}

# 提取每个节点的 CPD 数据
for node in model.nodes():
    cpd = model.get_cpds(node)
    values = cpd.values.flatten()
    cpd_data[node] = values

# 创建 DataFrame
cpd_df = pd.DataFrame(cpd_data)

# 打印 DataFrame
print(cpd_df)
