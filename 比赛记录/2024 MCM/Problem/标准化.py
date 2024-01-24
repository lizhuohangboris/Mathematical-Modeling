import pandas as pd
from sklearn.preprocessing import StandardScaler

# 加载数据集
data = pd.read_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/output.csv")  # 替换为实际文件路径

# 提取特征和标签
X = data.drop(['id', 'cardio'], axis=1)  # 通过传递一个列名列表来删除多列

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将标准化后的数据转为 DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 添加 'id' 和 'cardio' 列
X_scaled_df['id'] = data['id']
X_scaled_df['cardio'] = data['cardio']

# 将标准化后的数据保存为 CSV 文件
X_scaled_df.to_csv("C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/X_standardized_with_id_cardio.csv", index=False)
