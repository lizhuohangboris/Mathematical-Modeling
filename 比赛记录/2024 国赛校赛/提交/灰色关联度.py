import pandas as pd
import numpy as np

# 读取Excel文件
file_path = r'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/数据(1).xlsx'
df = pd.read_excel(file_path)

# 转置数据框
df_transposed = df.T

# 重命名列，使其更容易访问
df_transposed.columns = df_transposed.iloc[0]
df_transposed = df_transposed[1:]

# 定义需要的列
columns = ['GDP（十亿）', '人口（百万人）', '二氧化碳排放量（百万吨）', '能源消耗', '煤炭消耗量（百万吨）', '原油消耗量', '天然气消耗量', '新能源消耗量', '钢铁产量（千吨）', '水泥（百万吨）', '民用汽车数量（千辆）']

# 提取所需的列
data = df_transposed[columns]

# 将数据转换为数值类型
data = data.apply(pd.to_numeric, errors='coerce')

# 打印数据检查NaN
print("数据检查（如有NaN值）：")
print(data.isna().sum())

# 检查数据的基本统计信息
print("数据基本统计信息：")
print(data.describe())

# 参考序列（如二氧化碳排放量）
reference = data['二氧化碳排放量（百万吨）'].values

# 比较序列（如煤炭消耗量、GDP等）
comparisons = data.drop(columns=['二氧化碳排放量（百万吨）']).values.T

# 数据标准化（初值化法）
reference_standardized = reference / reference[0]
comparisons_standardized = comparisons / comparisons[:, 0][:, np.newaxis]

# 检查标准化后的数据
print("标准化后的参考序列：")
print(reference_standardized)
print("标准化后的比较序列：")
print(comparisons_standardized)

# 计算差异序列
diff = np.abs(comparisons_standardized - reference_standardized)

# 检查差异序列
print("差异序列：")
print(diff)

# 计算关联系数
rho = 0.5
diff_min = np.min(diff)
diff_max = np.max(diff)
correlation_coefficients = (diff_min + rho * diff_max) / (diff + rho * diff_max)

# 检查关联系数
print("关联系数：")
print(correlation_coefficients)

# 计算灰色关联度
grey_relations = np.mean(correlation_coefficients, axis=1)

# 输出结果
factors = data.columns.drop('二氧化碳排放量（百万吨）')
grey_relation_df = pd.DataFrame({
    '因素': factors,
    '灰色关联度': grey_relations
})

print(grey_relation_df)

# 按灰色关联度排序
grey_relation_df = grey_relation_df.sort_values(by='灰色关联度', ascending=False)

# 输出排序后的结果
print("按灰色关联度排序后的结果：")
print(grey_relation_df)
