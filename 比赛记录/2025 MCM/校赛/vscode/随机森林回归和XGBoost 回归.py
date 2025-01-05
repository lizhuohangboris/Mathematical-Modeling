from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 导入数据
file_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2025 MCM\校赛\数据集\整合.xlsx"
data = pd.read_excel(file_path)

# 2. 提取特征和目标变量
X = data[['elevation', 'bio1', 'bio12', 'bare land', 'road', 'grassland', 'tree', 'wood', 'shelter']]
y = data['abundance']  # 直接使用连续的目标变量

# 3. 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. 定义回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, random_state=42)

# 6. 训练模型
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# 7. 预测
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)

# 8. 评估模型
def evaluate_model(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.2f}")
    return rmse, r2

print("随机森林回归：")
evaluate_model(y_train, rf_train_pred, "训练集（RF）")
evaluate_model(y_test, rf_test_pred, "测试集（RF）")

print("\nXGBoost回归：")
evaluate_model(y_train, xgb_train_pred, "训练集（XGB）")
evaluate_model(y_test, xgb_test_pred, "测试集（XGB）")

# 9. 可视化实际值与预测值的对比
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.show()

plot_predictions(y_test, rf_test_pred, "Random Forest: Actual vs Predicted (Test Set)")
plot_predictions(y_test, xgb_test_pred, "XGBoost: Actual vs Predicted (Test Set)")
