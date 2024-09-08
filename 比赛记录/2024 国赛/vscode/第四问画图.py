import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib import font_manager

# 设置中文字体，假设系统中有SimHei字体（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 初始形状参数
alpha_prior = 1
beta_prior = 1

# 样本数
n_samples = 50

# 次品数量对应不同初始次品率的假设
defective_counts = [2, 5, 10]  # 对应5%、10%、20%的次品数量

# 次品率对应的情形标签
labels = ['5%', '10%', '20%']

# 设置颜色
colors = ['blue', 'orange', 'green']

# 创建图像
x = np.linspace(0, 1, 100)
plt.figure(figsize=(10, 6))

# 计算和绘制更新后的次品率分布
for i, defective_count in enumerate(defective_counts):
    # 更新后的形状参数
    alpha_posterior = alpha_prior + defective_count
    beta_posterior = beta_prior + n_samples - defective_count
    
    # 绘制Beta分布
    y = beta.pdf(x, alpha_posterior, beta_posterior)
    plt.plot(x, y, color=colors[i], label=f'原来的 {labels[i]} 次品率', linewidth=2)

# 图像修饰
plt.title('贝叶斯更新后的次品率分布', fontsize=16)
plt.xlabel('次品率', fontsize=14)
plt.ylabel('概率密度', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
