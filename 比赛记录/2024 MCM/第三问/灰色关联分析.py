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

# 计算每个列与 point_victorr 的灰色关联度
gray_correlations = df[interested_columns].apply(lambda x: x.corr(df['point_victorr'], method='spearman'), axis=0)

# 绝对值排序
sorted_gray_correlations = gray_correlations.abs().sort_values(ascending=False)

# 设置图形风格
sns.set(style="whitegrid")

# 创建一个条形图，展示灰色关联度结果
plt.figure(figsize=(12, 8))
sns.barplot(x=gray_correlations[sorted_gray_correlations.index].values, 
            y=sorted_gray_correlations.index, palette="viridis")

# 添加标题和标签
plt.title("Gray Correlation with Probability of Scoring")
plt.xlabel("Gray Correlation")
plt.ylabel("Features")

# 显示图形
plt.show()
