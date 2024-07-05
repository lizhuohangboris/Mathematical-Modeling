import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans

# 读取数据
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024亚太中文\train.csv'
data = pd.read_csv(file_path, encoding='gbk')

# 检查数据列名
print(data.columns)

# 如果没有“风险类别”列，则手动创建
if '风险类别' not in data.columns:
    # 使用 KMeans 聚类将洪水概率分为三类
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['风险类别'] = kmeans.fit_predict(data[['洪水概率']])

# 确认有“风险类别”和“洪水概率”列
print(data.columns)

# 移除目标变量和不需要的列
X = data.drop(['风险类别', '洪水概率'], axis=1)
y = data['风险类别']

# 计算各个指标对风险类别的互信息
mutual_info = mutual_info_classif(X, y, discrete_features='auto')

# 归一化权重
weights = mutual_info / np.sum(mutual_info)

# 打印各个指标的权重
weights_dict = dict(zip(X.columns, weights))
print("各指标权重：", weights_dict)

# 选取关键指标
selected_columns = ['季风强度', '地形排水', '河流管理', '森林砍伐', '城市化', '气候变化', '大坝质量', '淤积', '滑坡']
X_selected = data[selected_columns]

# 从提供的权重中选取对应的权重
selected_weights = np.array([weights_dict[col] for col in selected_columns])

# 建立风险评价模型
def risk_evaluation_model(data, weights):
    # 计算每个样本的风险得分
    risk_scores = np.dot(data, weights)
    return risk_scores

# 计算所有样本的风险得分
risk_scores = risk_evaluation_model(X_selected, selected_weights)

# 将风险得分添加到原数据中
data['风险得分'] = risk_scores

# 打印前几行数据
print(data.head())
