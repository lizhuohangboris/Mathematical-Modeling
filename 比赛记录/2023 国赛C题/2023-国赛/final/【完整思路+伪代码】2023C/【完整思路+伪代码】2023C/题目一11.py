import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 假设您有一个DataFrame df 包含销售数据，其中'Category'是蔬菜品类列，'Product'是单品列，'Sales'是销售量列
df = pd.read_csv('sales_data.csv')

# 计算不同品类之间的相关性
category_corr = df.groupby('Category')['Sales'].sum().reset_index()
sns.barplot(x='Category', y='Sales', data=category_corr)
plt.title('Sales by Category')
plt.xticks(rotation=45)
plt.show()

# 计算不同单品之间的相关性
product_corr = df.groupby('Product')['Sales'].sum().reset_index()
sns.barplot(x='Product', y='Sales', data=product_corr)
plt.title('Sales by Product')
plt.xticks(rotation=90)
plt.show()

# 计算品类和单品之间的相关性
category_product_corr = df.groupby(['Category', 'Product'])['Sales'].sum().reset_index()
pivot_table = category_product_corr.pivot_table(index='Category', columns='Product', values='Sales')
sns.heatmap(pivot_table, cmap='coolwarm', annot=True)
plt.title('Sales Correlation between Categories and Products')
plt.show()



#####相关性分析#########
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


#### 时间序列分析#######
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

# 假设您有一个DataFrame df 包含时间序列数据，其中'date'是日期列，'sales'是销售量列
df = pd.read_csv('sales_time_series.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 时间序列分解
result = seasonal_decompose(df['sales'], model='additive')
result.plot()
plt.show()

# 拟合ARIMA模型并预测
model = ARIMA(df['sales'], order=(1, 1, 1))
model_fit = model.fit(disp=0)
forecast = model_fit.forecast(steps=30)  # 预测未来30天的销售量



###聚类分析####
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 假设您有一个DataFrame df 包含要聚类的特征列，例如'sales_quantity'和'sales_revenue'
df = pd.read_csv('sales_data_for_clustering.csv')

# 特征标准化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 使用K均值聚类
kmeans = KMeans(n_clusters=3)  # 假设要分为3个簇
kmeans.fit(df_scaled)

# 添加簇标签到DataFrame
df['cluster_label'] = kmeans.labels_

# 聚类结果可视化
plt.scatter(df['sales_quantity'], df['sales_revenue'], c=df['cluster_label'], cmap='rainbow')
plt.xlabel('Sales Quantity')
plt.ylabel('Sales Revenue')
plt.title('Clustering Results')
plt.show()
