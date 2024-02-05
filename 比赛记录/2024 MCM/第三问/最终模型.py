import pandas as pd
import statsmodels.api as sm

# 读取Excel文件
file_path = r'C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\最后一场比赛.xlsx'
df = pd.read_excel(file_path)

# 选择自变量
independent_vars = ['server',  'p1_ace',  'p1_winner', 'p2_winner',
                         'p1_unf_err', 
                        'p1_net_pt_won', 'p2_net_pt_won', 'p1_break_pt', 'p1_break_pt_missed']

# 添加截距项
X = sm.add_constant(df[independent_vars])

# 构建多元线性回归模型
model = sm.OLS(df['point_victor'], X).fit()

# 打印回归系数
print(model.summary())
