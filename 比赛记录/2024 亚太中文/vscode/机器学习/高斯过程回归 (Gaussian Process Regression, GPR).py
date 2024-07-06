import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
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

# 使用更大的子集数据训练
subset_size = 5000  # 增加子集大小
subset_indices = np.random.choice(X_train.shape[0], subset_size, replace=False)
X_train_subset = X_train.iloc[subset_indices]
y_train_subset = y_train.iloc[subset_indices]

# 定义核函数 (Matern核)
kernel = C(1.0, (1e-4, 1e1)) * Matern(length_scale=1.0, nu=1.5)

# 初始化并训练高斯过程回归模型
gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
gpr_model.fit(X_train_subset, y_train_subset)

# 进行预测
y_pred_gpr, y_std = gpr_model.predict(X_test, return_std=True)

# 评估模型
mse_gpr = mean_squared_error(y_test, y_pred_gpr)
r2_gpr = r2_score(y_test, y_pred_gpr)

print(f'Mean Squared Error (GPR): {mse_gpr}')
print(f'R2 Score (GPR): {r2_gpr}')

# 输出预测不确定性
print("预测的标准差 (Prediction Standard Deviation):")
print(y_std)
