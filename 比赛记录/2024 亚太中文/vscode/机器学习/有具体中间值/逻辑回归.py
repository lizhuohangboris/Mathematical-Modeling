import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, log_loss

# 确认文件路径正确无误
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\第二问.csv'

# 加载数据，尝试使用逗号作为分隔符
data = pd.read_csv(file_path, delimiter=',', encoding='gbk')

# 查看列名，去除可能存在的空格
data.columns = data.columns.str.strip()

# 分离特征和目标变量
X = data.drop('洪水概率', axis=1)
y = data['洪水概率']

# 二值化目标变量
threshold = 0.5
y_binary = (y >= threshold).astype(int)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# 初始化并训练线性回归模型
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 进行预测
y_pred = linear_model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (Linear Regression): {mse}')
print(f'R2 Score (Linear Regression): {r2}')

# 初始化并训练逻辑回归模型
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_binary, y_train_binary)

# 预测概率
y_prob = logistic_model.predict_proba(X_test_binary)[:, 1]

# 线性组合
z = logistic_model.decision_function(X_test_binary)

# 预测标签
y_pred_logistic = logistic_model.predict(X_test_binary)

# 评估模型
mse_logistic = mean_squared_error(y_test_binary, y_pred_logistic)
r2_logistic = r2_score(y_test_binary, y_pred_logistic)
log_loss_value = log_loss(y_test_binary, y_prob)

print(f'Mean Squared Error (Logistic Regression): {mse_logistic}')
print(f'R2 Score (Logistic Regression): {r2_logistic}')
print(f'Log Loss (Logistic Regression): {log_loss_value}')

# 输出逻辑回归公式的系数和截距
intercept = logistic_model.intercept_[0]
coefficients = logistic_model.coef_[0]

print("逻辑回归方程:")
print(f"洪水概率 = 1 / (1 + exp(-({intercept:.4f}", end=' ')
for i, col in enumerate(X.columns):
    print(f"+ ({coefficients[i]:.4f} * {col})", end=' ')
print(")))")

# 输出中间量
print("预测概率 (Predicted Probabilities):")
print(y_prob)
print("线性组合 (Linear Combination):")
print(z)
