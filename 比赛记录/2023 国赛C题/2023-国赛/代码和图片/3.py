import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import seaborn as sns
import warnings

# 设置字体为宋体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

# 忽略警告信息
warnings.filterwarnings("ignore")

# 使用绝对文件路径读取Excel文件
file_path = r"C:\Users\92579\Desktop\final\蔬菜品类时序图.xlsx"
df_1_data1 = pd.read_excel(file_path)

# 创建三维散点图
fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')

# 提取蔬菜品类数据
vegetable_categories = df_1_data1.columns
x = np.arange(len(vegetable_categories))

# 为y轴生成一个等差数列，以便在三维图中将点分散开
y = np.arange(len(df_1_data1))
y, x = np.meshgrid(y, x)
y = y.flatten()
x = x.flatten()

# 提取关联系数作为z轴数据
z = df_1_data1.values.flatten()

# 创建散点图
scatter = ax.scatter(x, y, z, c=z, cmap='coolwarm', s=50, alpha=0.7)

# 设置轴标签
ax.set_xlabel('蔬菜品类', fontsize=12)
ax.set_ylabel('蔬菜品类', fontsize=12)
ax.set_zlabel('关联系数', fontsize=12)

# 调整标题位置
ax.title.set_position([0.5, 1.05])

# 添加颜色条
cbar = fig.colorbar(scatter)
cbar.set_label('关联系数', fontsize=12)

# 调整颜色条标签字体大小
cbar.ax.tick_params(labelsize=10)

# 设置标题
plt.title('蔬菜品类关联系数三维散点图', fontsize=14)

# 设置坐标刻度的字体大小
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='z', labelsize=10)

# 显示图形
plt.show()
