import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel("D:/Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 统计建模/随机森林.xlsx")

# 将"Month"列转换为日期类型
data['Month'] = pd.to_datetime(data['Month'], format='%b-%y')

# 将"Month"列拆分为年和月，并添加这两列作为特征
data['Year'] = data['Month'].dt.year
data['Month'] = data['Month'].dt.month

# 确定特征和目标列
X = data.drop(["AQI"], axis=1)
y = data["AQI"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost模型
model = XGBRegressor(n_estimators=100, random_state=42)

# 拟合模型
model.fit(X_train, y_train)

# 预测训练集和测试集的AQI值
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 计算模型在训练集和测试集上的R^2分数
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Train R^2 Score:", train_score)
print("Test R^2 Score:", test_score)

# 绘制预测值与真实值的关系图
plt.scatter(y_train, y_train_pred, color='blue', label='Train')
plt.scatter(y_test, y_test_pred, color='red', label='Test')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('True AQI')
plt.ylabel('Predicted AQI')
plt.title('True AQI vs Predicted AQI')
plt.legend()
plt.show()
