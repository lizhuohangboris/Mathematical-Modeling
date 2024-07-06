import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelBinarizer

# 确认文件路径正确无误
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\聚类后.csv'

# 加载数据，尝试使用逗号作为分隔符
data = pd.read_csv(file_path, delimiter=',', encoding='utf-8')

# 查看列名，去除可能存在的空格
data.columns = data.columns.str.strip()

# 分离特征和目标变量
X = data.drop('洪水概率', axis=1)
y = data['洪水概率']

# 将目标变量转换为分类标签
y = y.astype('category').cat.codes

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 设置参数网格
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_features': ['sqrt', 4],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2],
    'max_depth': [None, 4, 6],
    'criterion': ['gini']
}

# 初始化随机森林分类模型
rf = RandomForestClassifier(random_state=42)

# 使用随机搜索和交叉验证来找到最优参数
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=20, cv=5, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

# 输出最优参数
print("Best parameters found: ", random_search.best_params_)
best_model = random_search.best_estimator_

# 使用最优参数模型进行预测
y_pred = best_model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 绘制多分类ROC曲线
lb = LabelBinarizer()
y_test_binarized = lb.fit_transform(y_test)
y_proba = best_model.predict_proba(X_test)

for i in range(y_test_binarized.shape[1]):
    fpr, tpr, thresholds = roc_curve(y_test_binarized[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for class {i}')
    plt.legend(loc="lower right")
    plt.show()

# 绘制学习曲线
train_sizes, train_scores, test_scores = learning_curve(best_model, X, y, cv=5, n_jobs=-1, 
                                                        train_sizes=np.linspace(0.1, 1.0, 5))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")
plt.show()

# 绘制精确度-召回率曲线
y_scores = cross_val_predict(best_model, X, y, cv=3, method='predict_proba')

for i in range(y_test_binarized.shape[1]):
    precisions, recalls, thresholds = precision_recall_curve(y_test_binarized[:, i], y_scores[:, i])

    plt.figure()
    plt.plot(recalls, precisions, color='b', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for class {i}')
    plt.show()

# 灵敏度分析 - 特征重要性
importances = best_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# 输出特征的重要性
print("Feature ranking:")

for f in range(X.shape[1]):
    print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]})")

# 绘制特征重要性图表
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
