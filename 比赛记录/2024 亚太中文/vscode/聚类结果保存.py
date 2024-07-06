import pandas as pd

# 定义文件路径
input_file_path = r'D:\\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\第二问.csv'
output_file_path = r'D:\\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\聚类后.csv'

# 读取CSV文件
df = pd.read_csv(input_file_path, encoding='gbk')

# 定义分类函数
def categorize_probability(probability):
    if 0.285 <= probability <= 0.465:
        return '低'
    elif 0.470 <= probability <= 0.525:
        return '中'
    elif 0.530 <= probability <= 0.725:
        return '高'
    else:
        return '未知'

# 应用分类函数到洪水概率列
df['洪水概率'] = df['洪水概率'].apply(categorize_probability)

# 保存到新CSV文件
df.to_csv(output_file_path, index=False)

print(f"文件已保存到 {output_file_path}")
