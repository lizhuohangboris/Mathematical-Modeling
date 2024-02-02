import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 加载数据集
data = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/bayes_data.csv")

# 选择用于构建网络的特征
features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio','bmi-level','age-level','ap-level']
data = data[features]

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)


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
])


# 从训练集中估计参数
model.fit(train_data, estimator=MaximumLikelihoodEstimator)

for node in model.nodes():
    cpd = model.get_cpds(node)
    print(f"CPD for Node {node}:\n{cpd}")


# 在测试集上进行预测
predictions = model.predict(test_data.drop('cardio', axis=1))

# 计算准确度
accuracy = accuracy_score(test_data['cardio'], predictions['cardio'])
print(f"Accuracy: {accuracy}")

# 输出混淆矩阵、分类报告等
conf_matrix = confusion_matrix(test_data['cardio'], predictions['cardio'])
class_report = classification_report(test_data['cardio'], predictions['cardio'])

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# # 输出节点之间的结合关系
# for edge in model.edges():
#     print(f"Edge: {edge[0]} -> {edge[1]}")

# # 输出节点的父节点
# for node in model.nodes():
#     parents = model.get_parents(node)
#     print(f"Node: {node}, Parents: {parents}")
