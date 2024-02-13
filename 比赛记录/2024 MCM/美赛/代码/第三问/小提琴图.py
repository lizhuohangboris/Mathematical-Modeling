import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
file_path = r'C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\最后一场比赛.xlsx'
df = pd.read_excel(file_path)

# 感兴趣的列
interested_columns = ['server', 'serve_no',  'p1_ace', 'p2_ace', 'p1_winner', 'p2_winner',
                        'p1_double_fault', 'p2_double_fault', 'p1_unf_err', 'p2_unf_err',
                       'p1_net_pt', 'p2_net_pt', 'p1_net_pt_won', 'p2_net_pt_won', 'p1_break_pt', 'p2_break_pt',
                       'p1_break_pt_won', 'p2_break_pt_won', 'p1_break_pt_missed', 'p2_break_pt_missed',
                       'rally_count', 'speed_mph']

# 使用melt函数将数据变形
melted_df = pd.melt(df, id_vars=['point_victorr'], value_vars=interested_columns,
                    var_name='Feature', value_name='Value')

# 设置图形风格
sns.set(style="whitegrid")

# 创建小提琴图
plt.figure(figsize=(16, 10))
sns.violinplot(x='Feature', y='Value', hue='point_victorr', data=melted_df, palette="viridis", dodge=True)
plt.title("Violin Plot of Features Distribution by point_victorr")
plt.xticks(rotation=45, ha='right')  # 使x轴标签更易读
plt.show()
