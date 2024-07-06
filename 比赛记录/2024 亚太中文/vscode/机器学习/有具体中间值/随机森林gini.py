import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# 确认文件路径正确无误
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\第二问.csv'

# 加载数据，尝试使用逗号作为分隔符
data = pd.read_csv(file_path, delimiter=',', encoding='gbk')

# 查看列名，去除可能存在的空格
data.columns = data.columns.str.strip()

# 分离特征和目标变量
X = data.drop('洪水概率', axis=1)
y = data['洪水概率']

# 将y转化为分类变量，例如，二元分类
y = (y > 0.5).astype(int)  # 将阈值设为0.5，可以根据实际情况调整

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练随机森林分类模型
model = RandomForestClassifier(n_estimators=100, max_features=None, min_samples_split=2,
                              min_samples_leaf=1, criterion='gini', max_depth=None, random_state=42)
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 评估模型
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# 输出随机森林模型的参数
print(f'Number of Decision Trees: {model.n_estimators}')
print(f'Max Features per Tree: {model.max_features}')
print(f'Minimum Samples to Split a Node: {model.min_samples_split}')
print(f'Minimum Samples per Leaf Node: {model.min_samples_leaf}')
print(f'Criterion: {model.criterion}')
print(f'Maximum Depth of Trees: {model.max_depth}')

# # 绘制ROC曲线
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

# 绘制学习曲线
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, 
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
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(recalls, precisions, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
