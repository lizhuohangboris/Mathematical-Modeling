import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel("D:/Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 统计建模/tianjinproceed.xlsx")

# 将"Month"列转换为日期类型
data['Month'] = pd.to_datetime(data['Month'], format='%b-%y')

# 将"Month"列拆分为年和月，并添加这两列作为特征
data['Year'] = data['Month'].dt.year
data['Month'] = data['Month'].dt.month

# 将"质量等级"列进行标签编码
data['质量等级'] = data['质量等级'].astype('category').cat.codes

# 将"范围"列拆分成最小值和最大值两列
data[['范围_最小值', '范围_最大值']] = data['范围'].str.split('~', expand=True).astype(float)

# 删除原始的"范围"列
data = data.drop(["范围"], axis=1)

# 确定特征和目标列
X = data.drop(["AQI"], axis=1)
y = data["AQI"]

# 划分训练集和测试集
# 将时间序列数据按照时间顺序划分，避免数据泄露问题
split_index = int(len(data) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 拟合模型
model.fit(X_train, y_train)

# 计算模型在测试集上的R^2分数
score = model.score(X_test, y_test)
score2 = model.score(X_train, y_train)
print("R^2 Score:", score)
print("trian:R^2 Score:", score2)

# 可视化预测结果
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
plt.show()
