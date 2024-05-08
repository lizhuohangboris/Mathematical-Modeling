import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve


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

# 计算均方误差和平均绝对误差
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print("Train Mean Squared Error:", mse_train)
print("Train Mean Absolute Error:", mae_train)
print("Test Mean Squared Error:", mse_test)
print("Test Mean Absolute Error:", mae_test)

# 残差图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='green', marker='s', label='Test data')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差图')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=0, xmax=500, lw=2, color='red')
plt.xlim([0, 500])

# 预测分布图
plt.subplot(1, 2, 2)
plt.hist([y_train, y_test], bins=20, color=['blue', 'green'], alpha=0.5, label=['Training data', 'Test data'])
plt.xlabel('AQI值')
plt.ylabel('频数')
plt.title('预测分布图')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# 特征重要性图
plt.figure(figsize=(8, 5))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.xlabel('特征重要性')
plt.ylabel('特征')
plt.title('特征重要性图')
plt.show()



# 定义绘制学习曲线函数
def plot_learning_curve(estimator, X, y, cv=None, train_sizes=np.linspace(0.1, 1.0, 10)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Score")
    plt.xlabel("Training examples")
    plt.ylabel("Mean Squared Error")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# 绘制学习曲线
plot_learning_curve(model, X, y)
