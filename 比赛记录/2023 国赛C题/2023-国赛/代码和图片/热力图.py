import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties

# 假设有一些蔬菜品类数据
vegetable_categories = ['胡萝卜', '西兰花', '菠菜', '番茄', '黄瓜']
x_categories = len(vegetable_categories)
y_categories = len(vegetable_categories)

# 创建蔬菜品类之间的组合数据（示例）
X, Y = np.meshgrid(range(x_categories), range(y_categories))
# 假设Z值代表相关性，这里使用随机数据作为示例
Z = np.random.rand(y_categories, x_categories)

# 创建三维散点图
fig = plt.figure(figsize=(10, 6), dpi=300)
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图
scatter = ax.scatter(X, Y, Z, c=Z, cmap='coolwarm')

# 设置中文坐标轴标签和刻度
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)  # 替换为您自己的中文字体文件路径
ax.set_xlabel('蔬菜品类', fontproperties=font)
ax.set_ylabel('蔬菜品类', fontproperties=font)
ax.set_zlabel('相关性', fontproperties=font)

# 设置坐标轴刻度标签为蔬菜品类名称
ax.set_xticks(np.arange(x_categories))
ax.set_yticks(np.arange(y_categories))
ax.set_xticklabels(vegetable_categories, fontproperties=font)
ax.set_yticklabels(vegetable_categories, fontproperties=font)

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('颜色', fontproperties=font)

# 显示图形
plt.title('问题1花叶类的单品热力图', fontproperties=font)
plt.savefig('问题1花叶类的单品热力图.png', bbox_inches='tight')  # 保存图片
plt.show()
