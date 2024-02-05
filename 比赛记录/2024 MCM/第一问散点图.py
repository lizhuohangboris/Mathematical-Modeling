import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = r"C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\1 2球员整合.xlsx"
df = pd.read_excel(file_path)

# 根据条件分配颜色
colors = ['indianred' if score1 > score2 else 'c' for score1, score2 in zip(df['final_score1'], df['final_score2'])]

# 调整点的大小和透明度
plt.figure(figsize=(8, 6))
plt.scatter(df['final_score1'], df['final_score2'], c=colors, label='Performance Score 1 vs 2', marker='o', s=30, alpha=0.5)

plt.xlabel('Performance Score of Player1')
plt.ylabel('Performance Score of Player2')
plt.title('Scatter Plot of Performance Score of Player1 vs Player2')
plt.legend()
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 读取Excel文件
# file_path = r"C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\1 2球员整合.xlsx"
# df = pd.read_excel(file_path)

# # 设置颜色映射和样式
# colors = {0: 'lightgray', 1: 'deepskyblue', 2: 'salmon'}

# # 绘制柱状图
# sns.barplot(x='score1-2', y='point_victor', hue='vic', data=df, palette=colors)

# # 添加标签和标题
# plt.xlabel('score1-2')
# plt.ylabel('point_victor')
# plt.title('Bar Plot of score1-2 vs point_victor with vic color')

# # 显示图例
# plt.legend(title='vic')

# # 显示图形
# plt.show()
