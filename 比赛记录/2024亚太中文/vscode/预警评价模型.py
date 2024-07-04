import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取数据
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024亚太中文\train.csv'
data = pd.read_csv(file_path, encoding='gbk')

# 检查数据
print(data.head())

# 数据归一化处理
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data.drop(['洪水概率', '风险类别'], axis=1)), columns=data.columns[:-2])

# 选择关键指标
selected_columns = ['季风强度', '地形排水', '河流管理', '森林砍伐', '城市化', '气候变化', '大坝质量', '淤积', '滑坡']
X_selected = data[selected_columns]

from sklearn.feature_selection import mutual_info_classif

# 计算各个指标对风险类别的互信息
y = data['风险类别']
mutual_info = mutual_info_classif(X_selected, y, discrete_features='auto')

# 归一化权重
weights = mutual_info / np.sum(mutual_info)
weights_dict = dict(zip(selected_columns, weights))
print("各指标权重：", weights_dict)

import numpy as np

def risk_evaluation_model(data, weights):
    # 计算每个样本的风险得分
    risk_scores = np.dot(data, weights)
    return risk_scores

# 计算所有样本的风险得分
selected_weights = np.array([weights_dict[col] for col in selected_columns])
risk_scores = risk_evaluation_model(X_selected, selected_weights)

# 将风险得分添加到原数据中
data['风险得分'] = risk_scores
print(data[['洪水概率', '风险类别', '风险得分']].head())
