import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 加载数据
training_data_path = 'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问训练数据.xlsx'
prediction_data_path = 'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问2.xlsx'

training_data = pd.read_excel(training_data_path)
prediction_data = pd.read_excel(prediction_data_path)

# 分离特征和目标变量
X = training_data.drop(columns=['二氧化碳排放量（百万吨）'])
y = training_data['二氧化碳排放量（百万吨）']

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林回归模型
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 预测并计算均方误差
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# 使用训练好的模型进行预测
X_predict = prediction_data.drop(columns=['二氧化碳排放量（百万吨）'])
predicted_co2 = model.predict(X_predict)

# 将预测结果添加到预测数据中
prediction_data['二氧化碳排放量（百万吨）'] = predicted_co2

# 保存预测结果到新的Excel文件
output_path = 'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问预测结果_随机森林2.xlsx'
prediction_data.to_excel(output_path, index=False)

output_path, mse
