import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
import matplotlib.font_manager as fm

# 加载中文字体
zh_font_path = 'C:\Windows\Fonts\simhei.ttf'  # 请根据您的系统调整路径
zh_font = fm.FontProperties(fname=zh_font_path)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 读取CSV文件，指定编码为'gbk'
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\第二问.csv'
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

# 计算每个类别的区间（最小值和最大值）
grouped_intervals = data.groupby('风险类别')['洪水概率'].agg(['min', 'max']).reset_index()
grouped_intervals['区间'] = grouped_intervals['max'] - grouped_intervals['min']
print("每个类别的区间：")
print(grouped_intervals)

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

# 使用 Seaborn 样式
sns.set(style="whitegrid")

# 可视化部分
plt.figure(figsize=(12, 8))

# 散点图显示聚类结果
plt.subplot(2, 2, 1)
sns.scatterplot(x='洪水概率', y='风险得分', hue='风险类别', data=data, palette='Set1', s=50)
plt.title('聚类结果散点图', fontproperties=zh_font)
plt.xlabel('洪水概率', fontproperties=zh_font)
plt.ylabel('风险得分', fontproperties=zh_font)

# 聚类中心
plt.subplot(2, 2, 2)
sns.scatterplot(x=range(len(cluster_centers)), y=cluster_centers.flatten(), marker='x', color='red', s=100)
plt.title('聚类中心', fontproperties=zh_font)
plt.xlabel('中心编号', fontproperties=zh_font)
plt.ylabel('洪水概率', fontproperties=zh_font)

# 风险类别区间
plt.subplot(2, 2, 3)
sns.barplot(x='风险类别', y='区间', data=grouped_intervals, palette='Set1')
plt.title('风险类别区间', fontproperties=zh_font)
plt.xlabel('风险类别', fontproperties=zh_font)
plt.ylabel('区间 (最大值 - 最小值)', fontproperties=zh_font)

# 各指标权重条形图
plt.subplot(2, 2, 4)
sns.barplot(x=list(range(len(weights))), y=weights, palette='Set1')
plt.title('各指标权重', fontproperties=zh_font)
plt.xlabel('指标编号', fontproperties=zh_font)
plt.ylabel('权重', fontproperties=zh_font)

plt.tight_layout()
plt.show()

# 灵敏度分析结果条形图
plt.figure(figsize=(10, 4))
sns.barplot(x=list(range(len(sensitivities))), y=sensitivities, palette='Set1')
plt.title('灵敏度分析结果', fontproperties=zh_font)
plt.xlabel('指标编号', fontproperties=zh_font)
plt.ylabel('灵敏度', fontproperties=zh_font)
plt.show()
