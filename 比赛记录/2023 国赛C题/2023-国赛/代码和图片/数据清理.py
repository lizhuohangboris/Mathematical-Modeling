import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. 数据加载
# 假设您有一个CSV文件，使用pd.read_csv加载数据
df = pd.read_csv('sales_data.csv')

# 2. 数据预处理
# 2.1 数据清洗：删除重复值、处理异常值等
df.drop_duplicates(inplace=True)
df['Sales'] = df['Sales'].apply(lambda x: x if 0 <= x <= 1000 else None)

# 2.2 缺失值处理：删除包含缺失值的行或使用均值填充
df.dropna(inplace=True)
df['Sales'].fillna(df['Sales'].mean(), inplace=True)

# 2.3 特征工程：创建新特征或进行特征转换
df['Total_Sales'] = df['Category1'] + df['Category2']
df = pd.get_dummies(df, columns=['Category1', 'Category2'])

# 2.4 数据标准化：确保特征具有相同的尺度
scaler = StandardScaler()
df[['Category1', 'Category2', 'Total_Sales']] = scaler.fit_transform(df[['Category1', 'Category2', 'Total_Sales']])

# 3. 相关性分析
# 计算相关系数
correlation_matrix = df[['Category1', 'Category2']].corr()

# 4. 可视化相关性
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
