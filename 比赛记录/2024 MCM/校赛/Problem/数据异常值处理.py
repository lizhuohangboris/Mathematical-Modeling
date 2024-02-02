import pandas as pd

# 读取CSV文件
input_file_path = "C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/cardio_train.csv"
output_file_path = "C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/output.csv"

# 读取CSV文件，假设该文件中有ap_hi和ap_lo列
df = pd.read_csv(input_file_path)

# 筛选满足条件的行
filtered_df = df[df['ap_hi'] > df['ap_lo']]

# 将筛选结果写入新的CSV文件
filtered_df.to_csv(output_file_path, index=False)

# 打印输出结果
print("筛选后的数据已保存到:", output_file_path)
