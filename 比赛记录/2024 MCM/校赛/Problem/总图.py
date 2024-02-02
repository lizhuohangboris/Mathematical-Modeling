import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

# 从文件中读取数据
file_path = "C:/Users/92579/Desktop/MATH/2024-MCM/Problem/Problem/output.csv"
data = pd.read_csv(file_path)

# 数据划分
X = data[['age', 'height', 'weight', 'ap_hi', 'ap_lo']]
y = data['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# 数据标准化
scaler_visualization = StandardScaler()
X_train_scaled = scaler_visualization.fit_transform(X_train)

# 添加类别标签到训练集
X_train_scaled_with_label = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_train_scaled_with_label['cardio'] = y_train

# 使用 seaborn 的 pairplot 函数进行两两组合的可视化
sns.pairplot(X_train_scaled_with_label, hue='cardio', palette='coolwarm', markers=["o", "s"], plot_kws={'alpha':0.5})
plt.suptitle('Pairplot of Features with Cardio Class')
plt.show()
