import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 确认文件路径正确无误
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\第二问.csv'

# 加载数据，尝试使用逗号作为分隔符
data = pd.read_csv(file_path, delimiter=',', encoding='gbk')

# 查看列名，去除可能存在的空格
data.columns = data.columns.str.strip()

# 分离特征和目标变量
X = data.drop('洪水概率', axis=1)
y = data['洪水概率']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练K近邻回归模型
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# 进行预测
y_pred_knn = knn_model.predict(X_test)

# 评估模型
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print(f'Mean Squared Error (KNN): {mse_knn}')
print(f'R2 Score (KNN): {r2_knn}')

# 获取邻居信息
distances, indices = knn_model.kneighbors(X_test)

# 输出中间量
print("\n邻居索引 (Neighbor Indices):")
print(indices)

print("\n邻居距离 (Neighbor Distances):")
print(distances)

# 显示测试集中第一个样本的邻居及其对应的特征和目标值
sample_index = 0
print(f"\n测试集中第一个样本的邻居及其对应的特征和目标值 (Sample Index: {sample_index}):")
for neighbor_index in indices[sample_index]:
    print(f"邻居索引: {neighbor_index}")
    print(f"特征值: {X_train.iloc[neighbor_index].values}")
    print(f"目标值: {y_train.iloc[neighbor_index]}\n")
