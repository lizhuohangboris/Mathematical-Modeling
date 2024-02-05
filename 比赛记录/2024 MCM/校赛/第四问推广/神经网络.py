import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 读取Excel文件
file_path = r'C:\Users\92579\Documents\GitHub\Mathematical-Modeling\比赛记录\2024 MCM\美赛\2024_MCM-ICM_Problems\2024_MCM-ICM_Problems\最后一场比赛.xlsx'
df = pd.read_excel(file_path)

# 选择自变量和因变量
independent_vars = ['server', 'p1_ace', 'p1_winner', 'p2_winner',
                    'p1_unf_err', 'p1_net_pt_won', 'p2_net_pt_won', 'p1_break_pt', 'p1_break_pt_missed']
dependent_var = 'point_victorr'

# 划分训练集和测试集
X = df[independent_vars]
y = df[dependent_var]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建神经网络模型
model = Sequential()
model.add(Dense(units=10, input_dim=len(independent_vars), activation='relu'))
model.add(Dense(units=1, activation='linear'))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=0)

# 在测试集上进行预测
y_pred = model.predict(X_test_scaled)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)





# 获取测试集的实际值
actual_values = y_test.values

# 获取模型预测值
predictions = model.predict(X_test_scaled).flatten()

# 绘制实际 vs. 预测的散点图
plt.scatter(actual_values, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.title('Actual vs. Predicted')
plt.show()



# 计算残差
residuals = actual_values - predictions

# 绘制残差分布
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.show()

importances = model.feature_importances_

# 绘制特征重要性条形图
plt.bar(range(len(importances)), importances)
plt.xticks(range(len(importances)), feature_names, rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()
