import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 假设您有一个DataFrame df 包含销售数据，其中'Category1'和'Category2'是不同蔬菜品类的销售量列
df = pd.read_csv('sales_data.csv')

# 计算相关系数
correlation_matrix = df[['Category1', 'Category2']].corr()

# 可视化相关性
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()