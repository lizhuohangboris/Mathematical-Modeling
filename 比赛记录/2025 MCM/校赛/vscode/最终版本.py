from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
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

# 4. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. 定义回归模型
lin_reg = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, random_state=42)

# 6. 模型融合（Voting Regressor）
voting_reg = VotingRegressor(estimators=[
    ('linear', lin_reg),
    ('random_forest', rf_model),
    ('xgb', xgb_model)
])

# 7. 模型训练
voting_reg.fit(X_train, y_train)

# 8. 模型预测
y_pred_train = voting_reg.predict(X_train)
y_pred_test = voting_reg.predict(X_test)

# 9. 模型评估函数
def evaluate_model(y_true, y_pred, dataset_type):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{dataset_type} - RMSE: {rmse:.2f}, R²: {r2:.2f}")
    return rmse, r2

print("训练集评估：")
evaluate_model(y_train, y_pred_train, "训练集")

print("\n测试集评估：")
evaluate_model(y_test, y_pred_test, "测试集")

# 10. 可视化函数

# (1) 实际值 vs 预测值散点图
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.show()

plot_predictions(y_train, y_pred_train, "Training Set: Actual vs Predicted")
plot_predictions(y_test, y_pred_test, "Test Set: Actual vs Predicted")

# (2) 残差分析
residuals = y_test - y_pred_test

# 残差直方图
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20, color='purple', alpha=0.7, edgecolor='black')
plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
plt.xlabel('Residuals', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Residuals', fontsize=14)
plt.legend()
plt.show()

# 残差散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_test, residuals, alpha=0.7, color='green')
plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residuals vs Predicted Values', fontsize=14)
plt.legend()
plt.show()

# (3) 实际值 vs 预测值回归线
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred_test, ci=None, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title('Linear Fit: Actual vs Predicted', fontsize=14)
plt.show()

# (4) 学习曲线
train_sizes, train_scores, test_scores = learning_curve(
    voting_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, label="Training Error", color='blue')
plt.plot(train_sizes, test_scores_mean, label="Validation Error", color='red')
plt.xlabel("Training Set Size", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.title("Learning Curve", fontsize=14)
plt.legend()
plt.show()
