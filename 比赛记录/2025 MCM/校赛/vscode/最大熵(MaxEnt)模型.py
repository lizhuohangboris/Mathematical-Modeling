import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

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
    # 用均值填充缺失值（可以根据需要修改为其他填充策略）
    y.fillna(y.mean(), inplace=True)

# 3. 处理 X 中的缺失值
# 使用均值填充缺失值
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 4. 数据归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. 训练逻辑回归模型（最大熵模型）
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)

# 7. 预测
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

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
importance = model.coef_[0]  # 提取模型的系数
feature_importance = pd.DataFrame({'Feature': data[['elevation', 'bio1', 'bio12', 'desert', 'grassland', 'crop', 'tree', 
                                                    'wood', 'bare land', 'road', 'shelter']].columns, 
                                   'Importance': importance})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print("\n变量重要性：")
print(feature_importance)

# 10. 输出预测结果
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
results.to_excel(r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2025 MCM\校赛\数据集\预测结果.xlsx", index=False)
print("\n预测结果已保存至 '预测结果.xlsx'")



# 实际值与预测值的散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.7, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')  # 对角线
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title('Actual vs Predicted Values', fontsize=14)
plt.legend()
plt.show()

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