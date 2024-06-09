from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# 加载数据
file_path = 'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问训练数据.xlsx'
df = pd.read_excel(file_path)

# 特征和目标变量
features = ['年份', 'GDP（十亿）', '人口（百万人）', '能源消耗', '钢铁产量（千吨）', '水泥（百万吨）', 
            '民用汽车数量（千辆）', '煤炭消耗量（百万吨）', '原油消耗量', '天然气消耗量', '新能源消耗量']
target = '二氧化碳排放量（百万吨）'

X = df[features]
y = df[target]

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方误差 (MSE): {mse}")
print(f"R^2 值: {r2}")

# 显示模型系数
coefficients = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
coefficients
