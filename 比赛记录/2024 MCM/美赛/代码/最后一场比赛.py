# import pandas as pd
# import matplotlib.pyplot as plt

# # 读取Excel文件
# file_path = r"C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\最后一场比赛.xlsx"
# df = pd.read_excel(file_path)

# # 将elapsed_time列转换为时间格式
# df['elapsed_time'] = pd.to_datetime(df['elapsed_time'], format='%H:%M:%S')

# # 绘制随elapsed_time变化的final_score1和final_score2的可视化图
# plt.scatter(df['elapsed_time'], df['final_score1'], label='Final Score 1', color='olive', marker='o')
# plt.scatter(df['elapsed_time'], df['final_score2'], label='Final Score 2', color='darksalmon', marker='x')

# # 添加标签和图例
# plt.xlabel('Elapsed Time')
# plt.ylabel('Final Scores')
# plt.title('Final Scores over Elapsed Time')
# plt.legend()

# # 配置横轴时间轴
# plt.gca().xaxis_date()
# plt.gcf().autofmt_xdate()

# # 显示图形
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# # 读取Excel文件
# file_path = r"C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\最后一场比赛.xlsx"
# df = pd.read_excel(file_path)

# # 绘制score与final_score1的关系
# #plt.scatter(df['score'], df['final_score1'], label='Final Score 1', color='olive', marker='o')
# plt.plot(df['score'], df['final_score1'], label='Final Score 1', color='olive', marker='o')
# # 绘制score与final_score2的关系
# plt.plot(df['score'], df['final_score2'], label='Final Score 2', color='darksalmon', marker='o')

# # 添加标签和图例
# plt.xlabel('Score')
# plt.ylabel('Final Scores')
# plt.title('Final Scores over Score')
# plt.legend()

# # 显示图形
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = r"C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\最后一场比赛.xlsx"
df = pd.read_excel(file_path)

# 设置分数范围
min_score = 50  # 替换为您的最小分数
max_score = 150  # 替换为您的最大分数

# 筛选在分数范围内的数据
filtered_df = df[(df['score'] >= min_score) & (df['score'] <= max_score)]

# 绘制score与final_score1的关系，使用虚线
plt.plot(filtered_df['score'], filtered_df['final_score1'], label='Carlos Alcaraz', linestyle='--', color='gold', marker='o')
# 绘制score与final_score2的关系，使用虚线
plt.plot(filtered_df['score'], filtered_df['final_score2'], label='Novak Djokovic', linestyle='--', color='plum', marker='o')

# 添加垂直虚线
plt.axvline(x=min_score, color='gray', linestyle='-', label=f'Min Score: {min_score}')
plt.axvline(x=max_score, color='gray', linestyle='-', label=f'Max Score: {max_score}')

# 添加标签和图例
plt.xlabel('Score')
plt.ylabel('Final Scores')
plt.title('2023-wimbledon-1701 On-site Performance')
plt.legend()

# 显示图形
plt.show()
