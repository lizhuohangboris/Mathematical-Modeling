from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 导入数据
file_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2025 MCM\校赛\数据集\整合.xlsx"
data = pd.read_excel(file_path)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 假设 data 是你的完整数据集

# 原始数据 X 和 y
X = data[['elevation', 'bio1', 'bio12', 'bare land', 'road', 
          'grassland', 'tree', 'wood', 'shelter']]  # 特征
y = data['abundance']  # 目标变量：鸟类种群数量

# 拟合模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 模拟植物增减
X_simulation = X.copy()

# 模拟植物增加10%的覆盖度
X_simulation['tree'] = X_simulation['tree'] * 1.1  # 树木增加10%
X_simulation['grassland'] = X_simulation['grassland'] * 1.1  # 草地增加10%

# 模拟植物减少10%的覆盖度
X_simulation_decrease = X.copy()
X_simulation_decrease['tree'] = X_simulation_decrease['tree'] * 0.9  # 树木减少10%
X_simulation_decrease['grassland'] = X_simulation_decrease['grassland'] * 0.9  # 草地减少10%

# 进行预测
prediction_increase = model.predict(X_simulation)  # 模拟植物增多后的鸟类种群预测
prediction_decrease = model.predict(X_simulation_decrease)  # 模拟植物减少后的鸟类种群预测

# 输出模拟结果
print("模拟植物增加后的预测鸟类种群数量：", prediction_increase[:10])  # 输出前10个预测结果
print("模拟植物减少后的预测鸟类种群数量：", prediction_decrease[:10])  # 输出前10个预测结果
