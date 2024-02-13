import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 读取Excel文件
file_path = r'C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\最后一场比赛.xlsx'
df = pd.read_excel(file_path)

# 选择自变量和因变量
independent_vars = ['server', 'p1_ace', 'p1_winner', 'p2_winner',
                     'p1_unf_err', 'p1_net_pt_won', 'p2_net_pt_won', 'p1_break_pt', 'p1_break_pt_missed']
dependent_var = 'point_victor'

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[independent_vars], df[dependent_var], test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建线性回归模型
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test_scaled)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 打印回归系数
print('Intercept:', model.intercept_)
print('Coefficients:', dict(zip(independent_vars, model.coef_)))
