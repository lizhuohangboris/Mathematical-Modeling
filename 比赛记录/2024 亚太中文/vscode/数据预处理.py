import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 加载数据集
file_path =r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024亚太中文\train.csv'
data = pd.read_csv(file_path, encoding='gbk')

# 使用箱形图法处理异常值
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].apply(lambda x: x if lower_bound <= x <= upper_bound else (upper_bound if x > upper_bound else lower_bound))
    return df

# 对所有数值列处理异常值（跳过ID列）
for col in data.columns[1:]:
    data = remove_outliers(data, col)

# 初始化标准化器
scaler = StandardScaler()

# 保存原始数值列用于后续恢复
numerical_columns = data.columns[1:-1]  # 除去ID列和目标列
original_values = data[numerical_columns].copy()

# 对数值列进行标准化处理（除去ID列）
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# 划分数据集
X = data.iloc[:, 1:-1]  # 特征
y = data.iloc[:, -1]  # 目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 恢复原始数值列
data[numerical_columns] = original_values

# 生成处理后的CSV文件
processed_file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024亚太中文\train2.csv'  # 请将此路径替换为你希望保存文件的路径
data.to_csv(processed_file_path, index=False)

# 输出预处理后的数据和划分结果
print("预处理后的数据已保存至:", processed_file_path)

print("训练集和测试集的形状：")
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)
