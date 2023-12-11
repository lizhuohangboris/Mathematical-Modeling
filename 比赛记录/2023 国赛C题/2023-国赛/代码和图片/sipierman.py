#####相关性分析#########
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 假设您有一个DataFrame df 包含销售数据，其中'Category1'和'Category2'是不同蔬菜品类的销售量列
df = pd.read_csv('C:/Users/92579/Desktop/'+'sales_data.csv')



# 创建一个LabelEncoder对象
label_encoder = LabelEncoder()

# 将'Category'列中的字符串标签转换为数值标签
df['Product'] = label_encoder.fit_transform(df['Product'])

# 计算相关系数
correlation_matrix = df[['Sales', 'Product']].corr()


# 可视化相关性
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

