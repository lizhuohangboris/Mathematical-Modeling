import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei']  # 使用微软雅黑字体
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 文件路径
file_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 APMCM\数据\灰色关联度.xlsx"

# 读取Excel数据
data = pd.read_excel(file_path)

# 设置因变量和自变量
dependent_variables = ["Cat（万只）", "Dog（万只）", "宠物行业市场规模", "宠物食物市场规模"]
independent_variables = data.columns.difference(dependent_variables + ["Years"])

# 定义灰色关联度计算函数
def gray_relation_analysis(reference_cols, comparison_data):
    # 数据归一化
    normalized_data = (comparison_data - comparison_data.min()) / (comparison_data.max() - comparison_data.min())
    reference_data = normalized_data[reference_cols]  # 提取因变量数据
    other_columns = normalized_data.drop(columns=reference_cols)  # 剩余变量

    # 初始化结果字典
    gray_relation_results = pd.DataFrame(index=other_columns.columns, columns=reference_cols)

    # 分别计算每个因变量的灰色关联度
    for ref_col in reference_data.columns:
        ref_series = reference_data[ref_col]
        for col in other_columns.columns:
            diff = np.abs(ref_series - other_columns[col])  # 绝对差值
            min_diff = diff.min()
            max_diff = diff.max()
            relation = (min_diff + 0.5 * max_diff) / (diff + 0.5 * max_diff)  # 关联系数
            gray_relation_results.loc[col, ref_col] = relation.mean()  # 按列存储平均值

    # 返回灰色关联系数矩阵
    gray_relation_results = gray_relation_results.astype(float).reset_index()
    gray_relation_results.rename(columns={"index": "变量"}, inplace=True)
    return gray_relation_results

# 进行灰色关联度分析
gray_relation_results = gray_relation_analysis(dependent_variables, data)

# 打印结果
print("灰色关联度分析结果：")
print(gray_relation_results)

# 可视化灰色关联度（分因变量）
plt.figure(figsize=(16, 8))
for col in dependent_variables:
    plt.plot(gray_relation_results["变量"], gray_relation_results[col], marker='o', label=col)

plt.xlabel("自变量", fontsize=14)
plt.ylabel("灰色关联度", fontsize=14)
plt.title("各因变量与自变量的灰色关联度分析", fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# 保存结果到 Excel
output_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 APMCM\数据\Gray_Relation_Detail_Result.xlsx"
gray_relation_results.to_excel(output_path, index=False)
print(f"灰色关联度分析结果已保存至 {output_path}")
