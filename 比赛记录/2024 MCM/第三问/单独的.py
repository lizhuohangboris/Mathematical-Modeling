import pandas as pd

# 读取数据
file_path = r'C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\最后一场比赛.xlsx'
df = pd.read_excel(file_path)

# 感兴趣的列
interested_columns = ['server', 'serve_no', 'point_victor', 'p1_ace', 'p2_ace', 'p1_winner', 'p2_winner',
                        'p1_double_fault', 'p2_double_fault', 'p1_unf_err', 'p2_unf_err',
                       'p1_net_pt', 'p2_net_pt', 'p1_net_pt_won', 'p2_net_pt_won', 'p1_break_pt', 'p2_break_pt',
                       'p1_break_pt_won', 'p2_break_pt_won', 'p1_break_pt_missed', 'p2_break_pt_missed',
                       'rally_count', 'speed_mph','point_victorr']

# 计算每个列与 point_victorr 的相关性
correlations = df[interested_columns].corr(method='spearman')['point_victorr']

# 打印相关性结果
print("Correlation with point_victorr:")
print(correlations)
