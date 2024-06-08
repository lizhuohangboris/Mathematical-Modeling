import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取Excel文件
file_path = r'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/数据(1).xlsx'
df = pd.read_excel(file_path)
print("原始列名：", df.columns)

# 转置数据框
df_transposed = df.T

# 重命名列，使其更容易访问
df_transposed.columns = df_transposed.iloc[0]
df_transposed = df_transposed[1:]

# 打印转置后的列名
print("转置后的列名：", df_transposed.columns)

# 定义需要的列
columns = ['GDP（十亿）', '人口（百万人）', '二氧化碳排放量（百万吨）', '能源消耗', '煤炭消耗量（百万吨）', '原油消耗量', '天然气消耗量', '新能源消耗量', '钢铁产量（千吨）', '水泥（百万吨）', '民用汽车数量（千辆）']

# 检查列是否存在
missing_columns = [col for col in columns if col not in df_transposed.columns]
if missing_columns:
    print(f"以下列不存在于数据框中: {missing_columns}")

# 提取所需的列
data = df_transposed[columns]

# 将数据转换为数值类型
data = data.apply(pd.to_numeric, errors='coerce')

# 计算相关系数矩阵
correlation_matrix = data.corr()

# 输出相关系数矩阵
print("相关系数矩阵：")
print(correlation_matrix)

# 可视化相关系数矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('相关系数矩阵热图')
plt.show()

