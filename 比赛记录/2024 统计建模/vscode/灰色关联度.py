import pandas as pd
import numpy as np
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

# 步骤 1: 归一化处理
def normalize(df):
    return (df - df.min()) / (df.max() - df.min())

X_normalized = normalize(X)
y_normalized = normalize(y)

# 步骤 2: 计算灰色关联系数
def grey_relation_coefficient(X, y, rho=0.5):
    abs_diff = np.abs(X.sub(y, axis=0))  # 计算每个特征与目标的绝对差
    min_diff = abs_diff.min().min()  # 最小差
    max_diff = abs_diff.max().max()  # 最大差
    coeff = (min_diff + rho * max_diff) / (abs_diff + rho * max_diff)
    return coeff

grey_coefficients = grey_relation_coefficient(X_normalized, y_normalized)

# 步骤 3: 计算灰色关联度（每列与AQI的关联度）
grey_relation_degree = grey_coefficients.mean()

# 设置中文字体显示（如果使用 matplotlib，需安装中文字体包）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 绘制各个量与AQI的关联度柱状图
plt.figure(figsize=(10, 6))
grey_relation_degree.plot(kind='bar')
plt.title('各特征与AQI的灰色关联度', fontsize=16)
plt.ylabel('灰色关联度', fontsize=14)
plt.xlabel('特征', fontsize=14)
plt.xticks(rotation=45)
plt.show()
