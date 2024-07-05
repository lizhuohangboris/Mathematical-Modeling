import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取CSV文件，指定编码为'gbk'
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\train.csv'
data = pd.read_csv(file_path, encoding='gbk')

# 数据预处理：归一化处理
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# 设定聚类数目
n_clusters = 3

# 对洪水概率列进行聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['风险类别'] = kmeans.fit_predict(data[['洪水概率']])

# 计算各个指标对风险类别的互信息
X = data.drop(['风险类别', '洪水概率'], axis=1)
y = data['风险类别']
mutual_info = mutual_info_classif(X, y, discrete_features='auto')

# 打印各个指标的互信息值
print("各指标的互信息值：", mutual_info)

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

# 生成多次扰动权重后的灵敏度数据
def multiple_sensitivity_analysis(data, weights, delta=0.1, num_iterations=100):
    sensitivity_results = {feature: [] for feature in X.columns}
    for _ in range(num_iterations):
        for i in range(len(weights)):
            perturbed_weights = weights.copy()
            perturbed_weights[i] += delta
            perturbed_weights /= np.sum(perturbed_weights)
            perturbed_scores = risk_evaluation_model(data, perturbed_weights)
            sensitivity = np.mean(np.abs(perturbed_scores - risk_scores))
            sensitivity_results[X.columns[i]].append(sensitivity)
    return sensitivity_results

sensitivity_results = multiple_sensitivity_analysis(X, weights)

# 转换为DataFrame以便于可视化
sensitivity_df = pd.DataFrame(sensitivity_results)

# 绘制箱线图
plt.figure(figsize=(12, 8))
sns.boxplot(data=sensitivity_df, orient='h', palette='Set2')
plt.xlabel('敏感度')
plt.ylabel('特征')
plt.title('每个特征的灵敏度分析')
plt.show()

# 绘制热力图
plt.figure(figsize=(14, 10))
sns.heatmap(sensitivity_df.T, cmap='viridis', annot=True, cbar=True)
plt.xlabel('迭代次数')
plt.ylabel('特征')
plt.title('灵敏度分析热力图')
plt.show()

# 绘制瀑布图
def plot_waterfall(features, sensitivities):
    cumulative = 0
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (feature, sensitivity) in enumerate(zip(features, sensitivities)):
        ax.bar(i, sensitivity, bottom=cumulative, label=feature)
        cumulative += sensitivity
    ax.set_xlabel('特征')
    ax.set_ylabel('敏感度')
    ax.set_title('灵敏度分析瀑布图')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right')
    plt.show()

sorted_indices = np.argsort(sensitivities)[::-1]
sorted_sensitivities = np.array(sensitivities)[sorted_indices]
sorted_features = np.array(X.columns)[sorted_indices]

plot_waterfall(sorted_features, sorted_sensitivities)
