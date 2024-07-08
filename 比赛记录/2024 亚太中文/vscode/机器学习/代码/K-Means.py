import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif

# 读取CSV文件，指定编码为'gbk'
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\去掉id.csv'
data = pd.read_csv(file_path, encoding='gbk')

# 数据预处理：归一化处理
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# 设定聚类数目
n_clusters = 3

# 对洪水概率列进行聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['风险类别'] = kmeans.fit_predict(data[['洪水概率']])

# 聚类中心（即高、中、低风险类别的中心）
cluster_centers = kmeans.cluster_centers_
print("聚类中心：", cluster_centers)

# 按风险类别分组，计算每个组的平均值
grouped_data = data.groupby('风险类别').mean()

# 打印不同风险类别的指标特征
print(grouped_data)


# 计算各个指标对风险类别的互信息
X = data.drop(['风险类别', '洪水概率'], axis=1)
y = data['风险类别']
mutual_info = mutual_info_classif(X, y, discrete_features='auto')

# 归一化权重
weights = mutual_info / np.sum(mutual_info)

# 打印各个指标的权重
print("各指标权重：", weights)

def risk_evaluation_model(data, weights):
    # 计算每个样本的风险得分
    risk_scores = np.dot(data, weights)
    return risk_scores

# 计算所有样本的风险得分
risk_scores = risk_evaluation_model(X, weights)

# 将风险得分添加到原数据中
data['风险得分'] = risk_scores

def sensitivity_analysis(data, weights, delta=0.1):
    sensitivities = []
    for i in range(len(weights)):
        perturbed_weights = weights.copy()
        perturbed_weights[i] += delta
        perturbed_weights /= np.sum(perturbed_weights)
        perturbed_scores = risk_evaluation_model(data, perturbed_weights)
        sensitivity = np.mean(np.abs(perturbed_scores - risk_scores))
        sensitivities.append(sensitivity)
    return sensitivities

# 进行灵敏度分析
sensitivities = sensitivity_analysis(X, weights)

# 打印灵敏度分析结果
print("灵敏度分析结果：", sensitivities)
