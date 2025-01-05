from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 导入数据
file_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2025 MCM\校赛\数据集\整合.xlsx"
data = pd.read_excel(file_path)

# 2. 提取特征和目标变量
X = data[['elevation', 'bio1', 'bio12', 'bare land', 'road', 'grassland', 'tree', 'wood', 'shelter']]
y = data['abundance']  # 连续变量

# 3. 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 特征增强：多项式特征（degree=2）
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# 5. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# 6. 定义回归模型
lin_reg = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, random_state=42)

# 7. 使用Lasso（L1正则化）和Ridge（L2正则化）进行正则化
lasso_reg = Lasso(alpha=0.1)  # Lasso回归
ridge_reg = Ridge(alpha=1.0)  # Ridge回归

# 8. 模型融合（Voting Regressor）
voting_reg = VotingRegressor(estimators=[
    ('linear', lin_reg),
    ('random_forest', rf_model),
    ('xgb', xgb_model)
])

# 9. 超参数优化：使用GridSearchCV进行超参数优化
param_grid = {
    'random_forest__n_estimators': [50, 100, 200],
    'random_forest__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1, 0.3],
    'xgb__n_estimators': [50, 100, 200],
    'xgb__max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(estimator=voting_reg, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 输出最优的超参数
print("最优超参数：", grid_search.best_params_)

# 10. 模型训练
best_voting_reg = grid_search.best_estimator_
best_voting_reg.fit(X_train, y_train)

# 11. 模型预测
y_pred_train = best_voting_reg.predict(X_train)
y_pred_test = best_voting_reg.predict(X_test)

# 12. 模型评估函数
def evaluate_model(y_true, y_pred, dataset_type):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{dataset_type} - RMSE: {rmse:.2f}, R²: {r2:.2f}")
    return rmse, r2

print("训练集评估：")
evaluate_model(y_train, y_pred_train, "训练集")

print("\n测试集评估：")
evaluate_model(y_test, y_pred_test, "测试集")
