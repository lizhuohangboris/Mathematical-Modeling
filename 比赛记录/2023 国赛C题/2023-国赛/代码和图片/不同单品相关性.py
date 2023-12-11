import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

desktop_path = 'C:/Users/92579/Desktop/'  # 请将"YourUsername"替换为您的用户名称
csv_filename = 'coonected.csv'  # 您的CSV文件的文件名

# 构建完整的文件路径
csv_file_path = desktop_path + csv_filename

# 使用pandas读取CSV文件
df = pd.read_csv(csv_file_path)

# 计算不同单品之间的相关性
product_corr = df.groupby('Product')['Sales'].sum().reset_index()
sns.barplot(x='Product', y='Sales', data=product_corr)
plt.title('Sales by Product')
plt.xticks(rotation=90)
plt.show()