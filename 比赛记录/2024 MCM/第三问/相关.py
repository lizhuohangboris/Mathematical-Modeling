# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np


# # 读取Excel文件
# file_path = r'C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\最后一场比赛.xlsx'
# df = pd.read_excel(file_path)

# # 选择数值型的列
# numeric_columns = df.select_dtypes(include=[np.number])

# # 创建相关性矩阵
# correlation_matrix = numeric_columns.corr()

# # 使用Seaborn绘制热力图
# plt.figure(figsize=(15, 12))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
# plt.show()

# # 提取相关系数
# correlation_coefficients = correlation_matrix['point_victorr']

# # 打印相关系数
# print("Correlation Coefficients:")
# print(correlation_coefficients)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
file_path = r'C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\最后一场比赛.xlsx'
df = pd.read_excel(file_path)

# 感兴趣的列
interested_columns = ['server', 'serve_no', 'point_victor', 'p1_ace', 'p2_ace', 'p1_winner', 'p2_winner',
                        'p1_double_fault', 'p2_double_fault', 'p1_unf_err', 'p2_unf_err',
                       'p1_net_pt', 'p2_net_pt', 'p1_net_pt_won', 'p2_net_pt_won', 'p1_break_pt', 'p2_break_pt',
                       'p1_break_pt_won', 'p2_break_pt_won', 'p1_break_pt_missed', 'p2_break_pt_missed',
                       'rally_count', 'speed_mph']

# 提取子集
subset = df[interested_columns]

# 计算相关系数
correlation_matrix = subset.corr(method='spearman')

# 保存相关系数矩阵到文件
#correlation_matrix.to_csv(r'C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\correlation_matrix.csv', index=True)


# 打印相关系数矩阵
print("Correlation Matrix:")
print(correlation_matrix)

# 绘制相关系数热力图
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()
