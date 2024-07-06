import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# 确认文件路径正确无误
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\第二问.csv'

# 加载数据，尝试使用逗号作为分隔符
data = pd.read_csv(file_path, delimiter=',', encoding='gbk')

# 查看数据的前几行
print("数据前几行：")
print(data.head())

# 查看列名，去除可能存在的空格
data.columns = data.columns.str.strip()

# 查看处理后的列名
print("处理后的列名：")
print(data.columns)

# 分离特征和目标变量
X = data.drop('洪水概率', axis=1)
y = data['洪水概率']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 打印划分后的数据维度
print(f'训练集特征维度: {X_train.shape}')
print(f'测试集特征维度: {X_test.shape}')
print(f'训练集目标维度: {y_train.shape}')
print(f'测试集目标维度: {y_test.shape}')

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 打印标准化后的部分数据
print("标准化后的训练集特征前几行：")
print(X_train_scaled[:5])
print("标准化后的测试集特征前几行：")
print(X_test_scaled[:5])

# 初始化并训练XGBoost回归模型
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 进行预测
y_pred = model.predict(X_test_scaled)

# 打印预测值的前几行
print("预测值前几行：")
print(y_pred[:5])

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
