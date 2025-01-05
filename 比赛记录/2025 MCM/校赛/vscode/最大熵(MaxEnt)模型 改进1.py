import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import numpy as np

# 1. 导入数据
file_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2025 MCM\校赛\数据集\整合.xlsx"
data = pd.read_excel(file_path)

# 2. 提取特征和目标变量
X = data[['elevation', 'bio1', 'bio12', 'bare land', 'road', 'grassland', 'tree', 'wood', 'shelter']]
y = data['abundance']  # 连续目标变量

# 3. 将目标变量分箱（分类任务）
# 将 abundance 分为 "低丰度"、"中丰度"、"高丰度"
y_binned = pd.qcut(y, q=3, labels=["low", "medium", "high"])  # 分为三个类别

# 4. 处理缺失值
# 特征变量中的 NaN 使用均值填充
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 5. 特征归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binned, test_size=0.2, random_state=42)

# 7. 最大熵模型 + 超参数优化
# 使用 GridSearchCV 优化正则化参数 C
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
log_reg = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', random_state=42)
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 最优模型
best_log_reg = grid_search.best_estimator_

# 8. 模型评估
# 在训练集和测试集上预测分类结果
y_pred_train = best_log_reg.predict(X_train)
y_pred_test = best_log_reg.predict(X_test)

# 输出混淆矩阵和分类报告
print("训练集分类报告：")
print(classification_report(y_train, y_pred_train))

print("测试集分类报告：")
print(classification_report(y_test, y_pred_test))

print("混淆矩阵（测试集）：")
print(confusion_matrix(y_test, y_pred_test))

# 9. 输出最优参数和模型性能
print(f"最佳正则化参数 C：{grid_search.best_params_['C']}")
print(f"交叉验证最佳准确率：{grid_search.best_score_:.4f}")

# 10. 变量重要性
# 提取每个特征的系数
feature_importance = pd.DataFrame({
    'Feature': ['elevation', 'bio1', 'bio12', 'bare land', 'road', 'grassland', 'tree', 'wood', 'shelter'],
    'Coefficient': best_log_reg.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\n变量重要性：")
print(feature_importance)

# 11. 保存结果
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
results.to_excel(r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2025 MCM\校赛\数据集\最大熵模型预测结果.xlsx", index=False)
print("\n预测结果已保存至 '最大熵模型预测结果.xlsx'")
