import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor

# 读取数据集
data = pd.read_csv(r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\五个变量的数据.csv', encoding='gbk')

# 分离特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义基础模型
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
]

# 定义堆叠模型
stacking_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

# 训练堆叠模型
stacking_model.fit(X_train, y_train)

# 预测
y_pred = stacking_model.predict(X_test)

# 评估
print(f'Stacking Model R^2 Score: {stacking_model.score(X_test, y_test)}')
