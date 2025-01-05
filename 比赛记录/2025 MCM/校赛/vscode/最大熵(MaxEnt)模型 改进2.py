from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 导入数据
file_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2025 MCM\校赛\数据集\整合.xlsx"
data = pd.read_excel(file_path)

# 2. 提取特征和目标变量
X = data[['elevation', 'bio1', 'bio12', 'bare land', 'road', 'grassland', 'tree', 'wood', 'shelter']]
y = data['abundance']

# 3. 将目标变量分箱（优化分箱规则）
y_binned = pd.cut(y, bins=[0, 20, 50, np.inf], labels=["low", "medium", "high"])

# 4. 处理缺失值
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 5. 特征归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binned, test_size=0.2, random_state=42)

# 7. 特征选择
log_reg = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', class_weight='balanced')
selector = RFE(log_reg, n_features_to_select=5, step=1)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# 8. 超参数优化
param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1]}
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_selected, y_train)

# 最优模型
best_log_reg = grid_search.best_estimator_

# 9. 模型评估
y_pred_train = best_log_reg.predict(X_train_selected)
y_pred_test = best_log_reg.predict(X_test_selected)

print("训练集分类报告：")
print(classification_report(y_train, y_pred_train))
print("测试集分类报告：")
print(classification_report(y_test, y_pred_test))
print("混淆矩阵（测试集）：")
print(confusion_matrix(y_test, y_pred_test))

# 10. 变量重要性
selected_features = [col for col, selected in zip(['elevation', 'bio1', 'bio12', 'bare land', 'road', 
                                                   'grassland', 'tree', 'wood', 'shelter'], selector.support_) if selected]
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': best_log_reg.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\n变量重要性：")
print(feature_importance)

from sklearn.metrics import ConfusionMatrixDisplay

# 绘制混淆矩阵（测试集）
conf_matrix = confusion_matrix(y_test, y_pred_test, labels=["low", "medium", "high"])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["low", "medium", "high"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Test Set)", fontsize=14)
plt.show()