import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel("D:/Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 统计建模/随机森林.xlsx")

# 将日期列转换为日期时间类型并设置为索引
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# 准备特征和目标变量
X = data.drop(columns=["AQI"])
y = data["AQI"]

# 初始化时间序列分割器
tscv = TimeSeriesSplit(n_splits=5)

# 创建空列表来存储每个模型的预测结果
predictions = []

# 遍历每个时间序列分割，训练模型并进行预测
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 创建随机森林模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # 训练模型
    rf_model.fit(X_train, y_train)

    # 预测并存储结果
    y_pred = rf_model.predict(X_test)
    predictions.append(pd.Series(y_pred, index=X_test.index))

    # 计算并打印均方误差
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

# 将每次预测的结果合并为一个DataFrame
all_predictions = pd.concat(predictions)

# 绘制原始数据和预测结果的对比图
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["AQI"], label='Actual', color='blue', linewidth=2)
for i, pred in enumerate(predictions):
    plt.plot(pred.index, pred, label=f'Predicted {i}', linestyle='--')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.title('Actual vs. Predicted AQI')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
