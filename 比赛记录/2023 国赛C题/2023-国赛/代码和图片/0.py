import pandas as pd

# 读取附件1.xlsx和附件2.xlsx文件
df1 = pd.read_excel('C:/Users/92579/Desktop/final/附件1.xlsx')
df2 = pd.read_excel('C:/Users/92579/Desktop/final/附件2.xlsx')

# 使用merge函数将两个DataFrame根据单品编码合并
merged_df = pd.merge(df2, df1, on='单品编码', how='left')

# 保存合并后的数据为新的Excel文件conform.xlsx
merged_df.to_excel('conform.xlsx', index=False)