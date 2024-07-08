import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

# 确认文件路径正确无误
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\第二问.csv'

# 加载数据，尝试使用逗号作为分隔符
data = pd.read_csv(file_path, delimiter=',', encoding='gbk')

# 查看列名，去除可能存在的空格
data.columns = data.columns.str.strip()

# 分离特征和目标变量
X = data.drop('洪水概率', axis=1)
y = data['洪水概率']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化并训练LightGBM回归模型
model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 进行预测
y_pred = model.predict(X_test_scaled)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
