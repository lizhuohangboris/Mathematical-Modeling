import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
file_path = r'C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\最后一场比赛.xlsx'
df = pd.read_excel(file_path)

# 感兴趣的列
interested_columns = ['server',  'p1_ace',  'p1_winner', 'p2_winner',
                         'p1_unf_err', 
                        'p1_net_pt_won', 'p2_net_pt_won', 'p1_break_pt', 'p1_break_pt_missed', 'point_victorr']

# 创建成对图
sns.pairplot(df[interested_columns], hue='point_victorr', palette="viridis", height=3, corner=True)
plt.suptitle("Pair Plot of Features with point_victorr", y=1.02)
plt.show()