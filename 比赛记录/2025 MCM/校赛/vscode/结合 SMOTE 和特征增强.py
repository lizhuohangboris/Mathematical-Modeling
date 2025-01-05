from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# 1. 导入数据
file_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2025 MCM\校赛\数据集\整合.xlsx"
data = pd.read_excel(file_path)

# 2. 提取特征和目标变量
X = data[['elevation', 'bio1', 'bio12', 'bare land', 'road', 'grassland', 'tree', 'wood', 'shelter']]
y = data['abundance']

# 3. 将目标变量分箱
y_binned = pd.cut(y, bins=[0, 15, 40, np.inf], labels=["low", "medium", "high"])

# 4. 处理缺失值
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 5. 特征增强（交互特征 + 多项式特征）
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# 6. 特征归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# 7. 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binned, test_size=0.2, random_state=42)

# 8. 过采样（SMOTE）
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 9. 超参数优化
log_reg = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', class_weight='balanced')
param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1]}
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_resampled, y_train_resampled)

# 最优模型
best_log_reg = grid_search.best_estimator_

# 10. 模型评估
y_pred_train = best_log_reg.predict(X_train_resampled)
y_pred_test = best_log_reg.predict(X_test)

print("训练集分类报告：")
print(classification_report(y_train_resampled, y_pred_train))
print("测试集分类报告：")
print(classification_report(y_test, y_pred_test))
print("混淆矩阵（测试集）：")
print(confusion_matrix(y_test, y_pred_test))

# 11. 保存结果
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
results.to_excel(r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2025 MCM\校赛\数据集\最大熵模型优化预测结果.xlsx", index=False)
print("\n预测结果已保存至 '最大熵模型优化预测结果.xlsx'")
