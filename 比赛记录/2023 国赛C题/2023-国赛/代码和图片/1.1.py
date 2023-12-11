import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=12)  # 指定合适的字体文件路径

desktop_path = 'C:/Users/92579/Desktop/'  # 请将"YourUsername"替换为您的用户名称
csv_filename = 'coonected.csv'  # 您的CSV文件的文件名

# 构建完整的文件路径
csv_file_path = desktop_path + csv_filename

# 使用pandas读取CSV文件
df = pd.read_csv(csv_file_path)
# 假设您有一个DataFrame df 包含销售数据，其中'Category'是蔬菜品类列，'Product'是单品列，'Sales'是销售量列

# 计算不同品类之间的相关性
category_corr = df.groupby('Category')['Sales'].sum().reset_index()

# 绘制柱状图
plt.figure(figsize=(10, 6))  # 增加图形大小
sns.barplot(x='Category', y='Sales', data=category_corr, palette='Set2')  # 调整颜色和样式
plt.title('销售分类', fontproperties=font)  # 使用中文标题
plt.xticks(rotation=45, fontproperties=font)  # 使用中文刻度标签
plt.xlabel('产品分类', fontproperties=font)  # 添加坐标轴标签
plt.ylabel('销售额', fontproperties=font)  # 添加坐标轴标签
plt.gca().spines['top'].set_visible(False)  # 去除顶部边框
plt.gca().spines['right'].set_visible(False)  # 去除右侧边框

# 添加数据标签
for index, row in category_corr.iterrows():
    plt.text(index, row['Sales'], f'{row["Sales"]}', ha='center', va='bottom', fontproperties=font)

plt.show()
