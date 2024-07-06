import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 确认文件路径正确无误
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\八个.csv'

# 加载数据，尝试使用逗号作为分隔符
data = pd.read_csv(file_path, delimiter=',', encoding='gbk')

# 查看列名，去除可能存在的空格
data.columns = data.columns.str.strip()

# 分离特征和目标变量
X = data.drop('洪水概率', axis=1)
y = data['洪水概率']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# 输出回归方程
intercept = model.intercept_
coefficients = model.coef_

print("回归方程:")
print(f"洪水概率 = {intercept:.4f}", end=' ')
for i, col in enumerate(X.columns):
    print(f"+ ({coefficients[i]:.4f} * {col})", end=' ')
print()

# 灵敏度分析
sensitivity = {}
for i, col in enumerate(X.columns):
    sensitivity[col] = np.std(X_train[col]) * coefficients[i]

# 排序并打印灵敏度
sensitivity = {k: v for k, v in sorted(sensitivity.items(), key=lambda item: item[1], reverse=True)}
print("\n灵敏度分析结果（按影响排序）:")
for key, value in sensitivity.items():
    print(f"{key}: {value:.4f}")

# 可视化灵敏度分析
sns.barplot(x=list(sensitivity.keys()), y=list(sensitivity.values()))
plt.xticks(rotation=90)
plt.xlabel('特征')
plt.ylabel('灵敏度')
plt.title('特征灵敏度分析')
plt.show()
