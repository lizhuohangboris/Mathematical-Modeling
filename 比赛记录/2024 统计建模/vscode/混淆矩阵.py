import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体

# 读取数据
data = pd.read_excel("D:/Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 统计建模/随机森林.xlsx")

# 将"Month"列转换为日期类型
data['Month'] = pd.to_datetime(data['Month'], format='%b-%y')

# 将"Month"列拆分为年和月，并添加这两列作为特征
data['Year'] = data['Month'].dt.year
data['Month'] = data['Month'].dt.month

# 确定特征和目标列
X = data.drop(["AQI"], axis=1)
y = data["AQI"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 拟合模型
rf_model.fit(X_train, y_train)

# 预测训练集和测试集的AQI值
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

# 计算模型在训练集和测试集上的R²分数和MSE
train_score_rf = rf_model.score(X_train, y_train)
test_score_rf = rf_model.score(X_test, y_test)
train_mse_rf = mean_squared_error(y_train, y_train_pred_rf)
test_mse_rf = mean_squared_error(y_test, y_test_pred_rf)

print("Random Forest Train R^2 Score:", train_score_rf)
print("Random Forest Test R^2 Score:", test_score_rf)
print("Random Forest Train MSE Score:", train_mse_rf)
print("Random Forest Test MSE Score:", test_mse_rf)

# 绘制混淆矩阵
disp_rf = plot_confusion_matrix(rf_model, X_test, y_test, cmap=plt.cm.Blues, display_labels=['Low', 'Medium', 'High'])
disp_rf.ax_.set_title('Confusion Matrix - Random Forest')
plt.show()

# 绘制ROC曲线
fig, ax = plt.subplots()
rf_roc_disp = plot_roc_curve(rf_model, X_test, y_test, ax=ax)
plt.title('ROC Curve - Random Forest')
plt.show()

# 绘制学习曲线
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(rf_model, X, y, cv=5, n_jobs=-1,
                                                                      train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1])
plt.figure()
plt.title('Learning Curve - Random Forest')
plt.xlabel("Training examples")
plt.ylabel("Score")
train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")
plt.show()


# LightGBM模型
lgbm_model = LGBMRegressor(n_estimators=100, random_state=42)

# 拟合模型
lgbm_model.fit(X_train, y_train)

# 预测训练集和测试集的AQI值
y_train_pred_lgbm = lgbm_model.predict(X_train)
y_test_pred_lgbm = lgbm_model.predict(X_test)

# 计算模型在训练集和测试集上的R²分数和MSE
train_score_lgbm = lgbm_model.score(X_train, y_train)
test_score_lgbm = lgbm_model.score(X_test, y_test)
train_mse_lgbm = mean_squared_error(y_train, y_train_pred_lgbm)
test_mse_lgbm = mean_squared_error(y_test, y_test_pred_lgbm)

print("LightGBM Train R^2 Score:", train_score_lgbm)
print("LightGBM Test R^2 Score:", test_score_lgbm)
print("LightGBM Train MSE Score:", train_mse_lgbm)
print("LightGBM Test MSE Score:", test_mse_lgbm)

# 绘制混淆矩阵
disp_lgbm = plot_confusion_matrix(lgbm_model, X_test, y_test, cmap=plt.cm.Blues, display_labels=['Low', 'Medium', 'High'])
disp_lgbm.ax_.set_title('Confusion Matrix - LightGBM')
plt.show()

# 绘制ROC曲线
fig, ax = plt.subplots()
lgbm_roc_disp = plot_roc_curve(lgbm_model, X_test, y_test, ax=ax)
plt.title('ROC Curve - LightGBM')
plt.show()

# 绘制学习曲线
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(lgbm_model, X, y, cv=5, n_jobs=-1,
                                                                      train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1])
plt.figure()
plt.title('Learning Curve - LightGBM')
plt.xlabel("Training examples")
plt.ylabel("Score")
train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")
plt.show()


# XGBoost模型
xgb_model = XGBRegressor(n_estimators=100, random_state=42)

# 拟合模型
xgb_model.fit(X_train, y_train)

# 预测训练集和测试集的AQI值
y_train_pred_xgb = xgb_model.predict(X_train)
y_test_pred_xgb = xgb_model.predict(X_test)

# 计算模型在训练集和测试集上的R²分数和MSE
train_score_xgb = xgb_model.score(X_train, y_train)
test_score_xgb = xgb_model.score(X_test, y_test)
train_mse_xgb = mean_squared_error(y_train, y_train_pred_xgb)
test_mse_xgb = mean_squared_error(y_test, y_test_pred_xgb)

print("XGBoost Train R^2 Score:", train_score_xgb)
print("XGBoost Test R^2 Score:", test_score_xgb)
print("XGBoost Train MSE Score:", train_mse_xgb)
print("XGBoost Test MSE Score:", test_mse_xgb)

# 绘制混淆矩阵
disp_xgb = plot_confusion_matrix(xgb_model, X_test, y_test, cmap=plt.cm.Blues, display_labels=['Low', 'Medium', 'High'])
disp_xgb.ax_.set_title('Confusion Matrix - XGBoost')
plt.show()

# 绘制ROC曲线
fig, ax = plt.subplots()
xgb_roc_disp = plot_roc_curve(xgb_model, X_test, y_test, ax=ax)
plt.title('ROC Curve - XGBoost')
plt.show()

# 绘制学习曲线
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(xgb_model, X, y, cv=5, n_jobs=-1,
                                                                      train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1])
plt.figure()
plt.title('Learning Curve - XGBoost')
plt.xlabel("Training examples")
plt.ylabel("Score")
train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")
plt.show()
