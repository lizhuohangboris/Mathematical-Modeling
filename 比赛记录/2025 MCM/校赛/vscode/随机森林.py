from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 1. 导入数据
file_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2025 MCM\校赛\数据集\整合.xlsx"
data = pd.read_excel(file_path)

# 2. 提取特征和目标变量
X = data[['elevation', 'bio1', 'bio12', 'desert', 'grassland', 'crop', 'tree', 
          'wood', 'bare land', 'road', 'shelter']]
y = data['abundance']  # 或 'richness'，视目标分析任务而定

# 检查并处理目标变量 y 中的 NaN 值
if y.isnull().sum() > 0:
    print(f"目标变量 y 中存在 {y.isnull().sum()} 个缺失值，正在处理...")
    y.fillna(y.mean(), inplace=True)

# 3. 处理 X 中的缺失值
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 4. 数据归一化（随机森林对特征归一化不敏感，此步可选）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. 训练随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 7. 预测
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# 8. 评估模型
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"训练集 RMSE: {rmse_train}")
print(f"测试集 RMSE: {rmse_test}")
print(f"训练集 R²: {r2_train}")
print(f"测试集 R²: {r2_test}")

# 9. 变量重要性
importance = rf_model.feature_importances_  # 提取特征重要性
feature_importance = pd.DataFrame({'Feature': data[['elevation', 'bio1', 'bio12', 'desert', 'grassland', 'crop', 'tree', 
                                                    'wood', 'bare land', 'road', 'shelter']].columns, 
                                   'Importance': importance})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print("\n变量重要性：")
print(feature_importance)

# 10. 输出预测结果
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
results.to_excel(r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2025 MCM\校赛\数据集\随机森林预测结果.xlsx", index=False)
print("\n预测结果已保存至 '随机森林预测结果.xlsx'")
