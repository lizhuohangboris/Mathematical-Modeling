import pandas as pd

# 读取Excel文件
df = pd.read_excel('C:/Users/92579/Desktop/123.xlsx')

# 根据“单品类”分组并计算单位利润的平均值
average_profit = df.groupby('单品类')['单位利润'].mean().reset_index()

# 统计每种单品类的数量
count_per_category = df['单品类'].value_counts().reset_index()
count_per_category.columns = ['单品类', '数量']

# 将数量除以7并添加到average_profit DataFrame中
average_profit['数量/7'] = count_per_category['数量'] / 7

# 计算成本加成定价的平均值
average_cost_plus_pricing = df.groupby('单品类')['成本加成定价'].mean().reset_index()
average_cost_plus_pricing.columns = ['单品类', '成本加成定价平均值']

# 合并平均利润和成本加成定价平均值的数据
merged_df = pd.merge(average_profit, average_cost_plus_pricing, on='单品类')

# 根据单位利润列进行排序
merged_df_sorted = merged_df.sort_values(by='单位利润', ascending=False)

# 创建一个新的Excel文件并将结果保存到其中
with pd.ExcelWriter('average_profit_with_count2.xlsx', engine='xlsxwriter') as writer:
    merged_df_sorted.to_excel(writer, sheet_name='平均利润', index=False)
