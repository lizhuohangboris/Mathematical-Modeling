# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import numpy as np

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体

# # 读取数据
# data = pd.read_excel("D:/Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 统计建模/随机森林.xlsx")

# # 将"Month"列转换为日期类型
# data['Month'] = pd.to_datetime(data['Month'], format='%b-%y')

# # 将"Month"列拆分为年和月，并添加这两列作为特征
# data['Year'] = data['Month'].dt.year
# data['Month'] = data['Month'].dt.month

# # 确定特征和目标列
# X = data.drop(["AQI"], axis=1)
# y = data["AQI"]

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 随机森林模型
# model = RandomForestRegressor(n_estimators=150, min_samples_leaf=12, min_impurity_decrease=0.5, bootstrap=True, max_features=0.9, max_depth=3, random_state=42)

# # 输出随机森林参数值


# # 拟合模型
# model.fit(X_train, y_train)

# # 预测训练集和测试集的AQI值
# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test)

# # 计算模型在训练集和测试集上的R^2分数
# train_score = model.score(X_train, y_train)
# test_score = model.score(X_test, y_test)
# print("Train R^2 Score:", train_score)
# print("Test R^2 Score:", test_score)

# # 绘制预测值与真实值的关系图
# plt.scatter(y_train, y_train_pred, color='blue', label='Train')
# plt.scatter(y_test, y_test_pred, color='red', label='Test')
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# plt.xlabel('真实AQI')
# plt.ylabel('预测AQI')
# plt.title('[ 随机森林 ] 真实AQI vs 预测AQI')
# plt.legend()
# plt.show()




import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve
import numpy as np

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
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 拟合模型
model.fit(X_train, y_train)

# 预测训练集和测试集的AQI值
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 计算模型在训练集和测试集上的R^2分数
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Train R^2 Score:", train_score)
print("Test R^2 Score:", test_score)
print("estimator Maximum Number of Iterations:", model.n_estimators)
print("min_child_sample Minimum Amount of Data on a Leaf Node:", model.min_samples_leaf)
print("gamma Minimum Gain to be Added:", model.min_impurity_decrease)
print("subsample Sample Percentage:", model.bootstrap)
print("colsample Sample Proportion of the Variable:", model.max_features)
print("max_depth Depth of the Tree:", model.max_depth)
# 绘制预测值与真实值的关系图
plt.scatter(y_train, y_train_pred, color='blue', label='Train')
plt.scatter(y_test, y_test_pred, color='red', label='Test')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('真实AQI')
plt.ylabel('预测AQI')
plt.title('[ 随机森林 ] 真实AQI vs 预测AQI')
plt.legend()
plt.show()



# # 混淆矩阵
# def plot_confusion_matrix(y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title('Confusion Matrix')
#     plt.colorbar()
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     tick_marks = np.arange(len(np.unique(y_true)))
#     plt.xticks(tick_marks, np.unique(y_true))
#     plt.yticks(tick_marks, np.unique(y_true))
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], 'd'),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.show()

# ROC曲线
# def plot_roc_curve(y_true, y_pred):
#     fpr, tpr, thresholds = roc_curve(y_true, y_pred)
#     plt.plot(fpr, tpr, label='ROC Curve')
#     plt.plot([0, 1], [0, 1], 'k--', label='Random')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.legend()
#     plt.show()

# 学习曲线
def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

# 使用以上定义的函数进行绘图
# plot_confusion_matrix(y_test, y_test_pred)
# plot_roc_curve(y_test, y_test_pred)
plot_learning_curve(model, X_train, y_train)
