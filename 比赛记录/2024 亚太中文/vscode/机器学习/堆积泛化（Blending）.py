import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# 读取数据集
data = pd.read_csv(r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\第二问.csv', encoding='gbk')

# 分离特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义基础模型
base_models = [
    RandomForestRegressor(n_estimators=100, random_state=42),
    GradientBoostingRegressor(n_estimators=100, random_state=42)
]

# 训练基础模型并收集预测结果
train_preds = np.zeros((X_train.shape[0], len(base_models)))
test_preds = np.zeros((X_test.shape[0], len(base_models)))

for i, model in enumerate(base_models):
    model.fit(X_train, y_train)
    train_preds[:, i] = model.predict(X_train)
    test_preds[:, i] = model.predict(X_test)

# 使用基础模型的预测结果训练元模型
blender = LinearRegression()
blender.fit(train_preds, y_train)

# 预测
y_pred_blend = blender.predict(test_preds)

# 评估
print(f'Blending Model R^2 Score: {r2_score(y_test, y_pred_blend)}')
