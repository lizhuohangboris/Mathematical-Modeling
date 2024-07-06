import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

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

# 线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f'Linear Regression - Mean Squared Error: {mse_lr}')
print(f'Linear Regression - R2 Score: {r2_lr}')

# 决策树回归模型
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f'Decision Tree Regressor - Mean Squared Error: {mse_dt}')
print(f'Decision Tree Regressor - R2 Score: {r2_dt}')
print(f'Decision Tree Regressor - Tree Depth: {dt_model.get_depth()}')
print(f'Decision Tree Regressor - Number of Leaves: {dt_model.get_n_leaves()}')

# 支持向量回归模型
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)
print(f'Support Vector Regressor - Mean Squared Error: {mse_svr}')
print(f'Support Vector Regressor - R2 Score: {r2_svr}')

# 随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest Regressor - Mean Squared Error: {mse_rf}')
print(f'Random Forest Regressor - R2 Score: {r2_rf}')
print(f'Random Forest Regressor - Number of Trees: {len(rf_model.estimators_)}')
print(f'Random Forest Regressor - Feature Importances: {rf_model.feature_importances_}')

# XGBoost回归模型
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f'XGBoost Regressor - Mean Squared Error: {mse_xgb}')
print(f'XGBoost Regressor - R2 Score: {r2_xgb}')
print(f'XGBoost Regressor - Number of Trees: {xgb_model.n_estimators}')
print(f'XGBoost Regressor - Feature Importances: {xgb_model.feature_importances_}')

# LightGBM回归模型
lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)
print(f'LightGBM Regressor - Mean Squared Error: {mse_lgb}')
print(f'LightGBM Regressor - R2 Score: {r2_lgb}')
print(f'LightGBM Regressor - Number of Trees: {lgb_model.n_estimators}')
print(f'LightGBM Regressor - Feature Importances: {lgb_model.feature_importances_}')
