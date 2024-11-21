import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 文件路径
file_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 APMCM\数据\1.xlsx"

# 读取Excel文件
data = pd.read_excel(file_path)

# 确保删除非数值列，例如年份或其他说明列
data_numeric = data.select_dtypes(include=['float64', 'int64'])

# 检查是否有 NaN 值
if data_numeric.isnull().any().any():
    print("数据中存在 NaN 值，正在进行处理...")
    # 方法1: 删除包含 NaN 值的行
    # data_numeric = data_numeric.dropna()

    # 方法2: 用均值填充缺失值
    data_numeric = data_numeric.fillna(data_numeric.mean())

# 标准化数据
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# PCA分析
pca = PCA()
pca_result = pca.fit_transform(data_scaled)

# 主成分解释方差比
explained_variance = pd.DataFrame({
    "Principal Component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "Explained Variance Ratio": pca.explained_variance_ratio_
})

# 主成分载荷矩阵
pca_components = pd.DataFrame(
    pca.components_.T,
    index=data_numeric.columns,
    columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])]
)

# 保存结果到Excel
output_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 APMCM\数据\PCA_Result.xlsx"
with pd.ExcelWriter(output_path) as writer:
    explained_variance.to_excel(writer, sheet_name="Explained Variance", index=False)
    pca_components.to_excel(writer, sheet_name="Principal Component Loadings")

print(f"主成分分析结果已保存至 {output_path}")

# 查看第一主成分的载荷值
pc1_loadings = pca_components["PC1"]

# 排序载荷值（绝对值最大表示影响大）
pc1_loadings_sorted = pc1_loadings.abs().sort_values(ascending=False)

print("PC1 对变量的贡献（按绝对值排序）：")
print(pc1_loadings_sorted)
