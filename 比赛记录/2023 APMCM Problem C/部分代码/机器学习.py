import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data=pd.read_csv('Q1.csv')

# 分割自变量和目标变量
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
# 训练模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
# 预测结果
y_pred = rf.predict(X_test)
 
# 计算MSE和R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
 
# 输出模型评估结果和目标方程
print('MSE:', mse)
print('R-squared:', r2)
# 绘制特征重要性条形图
feature_importance = rf.feature_importances_
feature_names = X.columns.tolist()
sorted_idx = feature_importance.argsort()
 
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.barh(range(len(feature_importance)), feature_importance[sorted_idx])
plt.yticks(range(len(feature_importance)), [feature_names[i] for i in sorted_idx],fontsize=10)
plt.xlabel('特征重要性')
plt.ylabel('特征名称')
plt.title('随机森林回归特征重要性')
plt.savefig('随机森林回归特征重要性',dpi=300)
