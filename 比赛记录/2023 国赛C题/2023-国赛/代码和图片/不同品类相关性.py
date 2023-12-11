import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

desktop_path = 'C:/Users/92579/Desktop/'  # 请将"YourUsername"替换为您的用户名称
csv_filename = 'coonected.csv'  # 您的CSV文件的文件名

# 构建完整的文件路径
csv_file_path = desktop_path + csv_filename

# 使用pandas读取CSV文件
df = pd.read_csv(csv_file_path)

# 假设您有一个DataFrame df 包含销售数据，其中'Category'是蔬菜品类列，'Product'是单品列，'Sales'是销售量列
# df = pd.read_csv('sales_data.csv')

# 计算不同品类之间的相关性
category_corr = df.groupby('分类名称')['销量(千克)'].sum().reset_index()
sns.barplot(x='分类名称', y='销量(千克)', data=category_corr)
plt.title('销量(千克) by 分类名称')
plt.xticks(rotation=45)
plt.show()
