import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 从文件中读取数据
file_path = "C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/output.csv"
data = pd.read_csv(file_path)

# 数据划分
X = data[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']]
y = data['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# 选择两个特征进行可视化
X_visualization = X_train[['ap_hi', 'ap_lo']]  # 请根据实际需求选择两个特征

# 数据标准化
scaler_visualization = StandardScaler()
X_visualization_scaled = scaler_visualization.fit_transform(X_visualization)

# 训练逻辑回归模型
model_visualization = LogisticRegression(random_state=42)
model_visualization.fit(X_visualization_scaled, y_train)

# 生成一系列测试数据
x_min, x_max = X_visualization_scaled[:, 0].min() - 1, X_visualization_scaled[:, 0].max() + 1
y_min, y_max = X_visualization_scaled[:, 1].min() - 1, X_visualization_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model_visualization.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.3)

# 绘制数据点
plt.scatter(X_visualization_scaled[:, 0], X_visualization_scaled[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', s=10)

# 修改图标题
plt.title('Logistic Regression Fit (Visualization)\nAp_hi vs. Ap_lo')

plt.xlabel('Ap_hi (Scaled)')
plt.ylabel('Ap_lo (Scaled)')
plt.show()
