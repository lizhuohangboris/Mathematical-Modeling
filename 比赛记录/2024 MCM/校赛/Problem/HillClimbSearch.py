import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pgmpy.estimators import MaximumLikelihoodEstimator

# 加载数据集
data = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/cardio_train.csv")

# 选择用于构建网络的特征
features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
data = data[features]

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 使用 HillClimbSearch 进行结构学习
hc = HillClimbSearch(train_data, scoring_method=BicScore(train_data))
best_model = hc.estimate()
print("Edges in the best model:", best_model.edges())

# 创建贝叶斯网络模型
model = BayesianModel(best_model.edges())

# 从训练集中估计参数
model.fit(train_data, estimator=MaximumLikelihoodEstimator)

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
