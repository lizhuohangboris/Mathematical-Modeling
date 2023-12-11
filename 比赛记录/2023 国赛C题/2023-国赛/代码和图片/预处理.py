import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 加载数据
data = pd.read_csv('C:/Users/92579/Desktop/coonected.csv')

# 2. 缺失值处理（用每列的均值填充缺失值）
data.fillna(data.mean(), inplace=True)

# 3. 数据归一化
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# 4. 数据标准化
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# 5. 保存预处理后的数据到新的CSV文件
pd.DataFrame(data_normalized, columns=data.columns).to_csv('yuchuli1.csv', index=False)
pd.DataFrame(data_standardized, columns=data.columns).to_csv('yuchuli2.csv', index=False)
