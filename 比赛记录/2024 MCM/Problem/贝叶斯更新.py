import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pgmpy.estimators import ParameterEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import ParameterEstimator

# 加载数据集
data = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/bayes_data.csv")

# 创建新的列，表示生理指标、医疗指标、主观信息
data['生理指标'] = data['gender'] + data['age-level'] + data['bmi-level']
data['医疗指标'] = data['ap-level'] + data['cholesterol'] + data['gluc']
data['主观信息'] = data['smoke'] + data['alco'] + data['active']

# 选择用于构建网络的特征，包括新创建的列
features = ['生理指标', '医疗指标', '主观信息', 'cardio']
data = data[features]

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.7, random_state=42)

# 创建新的贝叶斯网络模型
new_model = BayesianModel([
    ('生理指标', 'cardio'),
    ('医疗指标', 'cardio'),
    ('主观信息', 'cardio'),
])

# 在训练集上估计概率分布
for node in new_model.nodes():
    if node != 'cardio':
        cpd = MaximumLikelihoodEstimator(new_model, train_data).estimate_cpd(node)
        new_model.add_cpds(cpd)

# 单独处理 'cardio' 节点
cpd_cardio = MaximumLikelihoodEstimator(new_model, train_data).estimate_cpd('cardio')
new_model.add_cpds(cpd_cardio)




# 打印新结构的 CPDs
for node in new_model.nodes():
    cpd = new_model.get_cpds(node)
    print(f"CPD for Node {node}:\n{cpd}")

# 在测试集上进行预测
new_predictions = new_model.predict(test_data.drop('cardio', axis=1))

# 计算准确度
new_accuracy = accuracy_score(test_data['cardio'], new_predictions['cardio'])
print(f"Accuracy for the new model: {new_accuracy}")

# 输出混淆矩阵、分类报告等
conf_matrix = confusion_matrix(test_data['cardio'], new_predictions['cardio'])
class_report = classification_report(test_data['cardio'], new_predictions['cardio'])

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# 输出节点之间的结合关系
for edge in new_model.edges():
    print(f"Edge: {edge[0]} -> {edge[1]}")

# 输出节点的父节点
for node in new_model.nodes():
    parents = new_model.get_parents(node)
    print(f"Node: {node}, Parents: {parents}")
