import pandas as pd

# 读取数据
file_path = r'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/数据(2).xlsx'
data = pd.read_excel(file_path)

# 对数据进行二阶差分处理
data_diff = data.diff().diff().dropna()

# 保存差分后的数据
output_file_path = r'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/数据_difff.xlsx'
data_diff.to_excel(output_file_path, index=False)
