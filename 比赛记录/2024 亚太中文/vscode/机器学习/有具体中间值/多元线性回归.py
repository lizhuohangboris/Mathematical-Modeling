import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
print(f"洪水概率 = {intercept:.10f}", end=' ')
for i, col in enumerate(X.columns):
    print(f"+ ({coefficients[i]:.10f} * {col})", end=' ')
print()

# 设置中文字体
font = FontProperties(fname='C:\\Windows\\Fonts\\simhei.ttf')

# 可视化实际值和预测值的关系
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('实际值', fontproperties=font)
plt.ylabel('预测值', fontproperties=font)
plt.title('实际值 vs 预测值', fontproperties=font)
plt.show()

# 可视化残差分析
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, edgecolors=(0, 0, 0))
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='dashed')
plt.xlabel('预测值', fontproperties=font)
plt.ylabel('残差', fontproperties=font)
plt.title('残差分析', fontproperties=font)
plt.show()

# 可视化特征的系数
plt.figure(figsize=(12, 6))
plt.barh(X.columns, coefficients)
plt.xlabel('系数值', fontproperties=font)
plt.ylabel('特征', fontproperties=font)
plt.title('多元线性回归中各特征的系数', fontproperties=font)
plt.show()
