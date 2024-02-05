# import pandas as pd

# # 读取Excel文件
# file_path = r"C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\最后一场比赛.xlsx"
# df = pd.read_excel(file_path)

# # 计算 Pearson 和 Spearman 相关性系数
# pearson_corr_final_score1 = df['point_victorr'].corr(df['final_score1'], method='pearson')
# spearman_corr_final_score1 = df['point_victorr'].corr(df['final_score1'], method='spearman')

# pearson_corr_final_score2 = df['point_victorr'].corr(df['final_score2'], method='pearson')
# spearman_corr_final_score2 = df['point_victorr'].corr(df['final_score2'], method='spearman')

# pearson_corr_score1_2 = df['point_victorr'].corr(df['score1-2'], method='pearson')
# spearman_corr_score1_2 = df['point_victorr'].corr(df['score1-2'], method='spearman')

# print("Pearson Correlation Coefficient (final_score1):", pearson_corr_final_score1)
# print("Spearman Correlation Coefficient (final_score1):", spearman_corr_final_score1)

# print("Pearson Correlation Coefficient (final_score2):", pearson_corr_final_score2)
# print("Spearman Correlation Coefficient (final_score2):", spearman_corr_final_score2)

# print("Pearson Correlation Coefficient (score1-2):", pearson_corr_score1_2)
# print("Spearman Correlation Coefficient (score1-2):", spearman_corr_score1_2)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = r"C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\1 2球员整合.xlsx"
df = pd.read_excel(file_path)

# 计算相关性矩阵
# correlation_matrix = df[['point_victor', 'Performance Score(P1)', 'Performance Score(P2)', 'Performance Score(P1-P2)']].corr()
correlation_matrix = df[['point_victor', 'final_score1', 'final_score2', 'score1-2']].corr()

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

