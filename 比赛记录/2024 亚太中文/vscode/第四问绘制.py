import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import numpy as np
import matplotlib.font_manager as fm

# 设置中文字体
font_path = 'C:/Windows/Fonts/simhei.ttf'  # 这是黑体字体路径，根据您的系统字体路径调整
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# 读取submit.csv文件，指定编码为gbk（如果您知道是其他编码，请更改为相应编码）
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\submit.csv'
data = pd.read_csv(file_path, encoding='gbk')

# 提取洪水概率数据
flood_probabilities = data['洪水概率']

# 绘制直方图和正态分布曲线
plt.figure(figsize=(12, 6))
sns.histplot(flood_probabilities, bins=50, kde=True, stat="density", linewidth=0)
mu, std = norm.fit(flood_probabilities)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label=f'正态分布\n$\mu={mu:.2f}$, $\sigma={std:.2f}$')
plt.xlabel('洪水概率')
plt.ylabel('频率')
plt.title('洪水概率的直方图和正态分布曲线')
plt.legend()
plt.show()

# 箱线图
plt.figure(figsize=(12, 6))
sns.boxplot(y=flood_probabilities)
plt.xlabel('洪水概率')
plt.title('洪水概率的箱线图')
plt.show()

# 小提琴图
plt.figure(figsize=(12, 6))
sns.violinplot(y=flood_probabilities)
plt.xlabel('洪水概率')
plt.title('洪水概率的小提琴图')
plt.show()

# 核密度估计图
plt.figure(figsize=(12, 6))
sns.kdeplot(flood_probabilities, shade=True)
plt.xlabel('洪水概率')
plt.title('洪水概率的核密度估计图')
plt.show()

# 累积分布函数图
plt.figure(figsize=(12, 6))
sns.ecdfplot(flood_probabilities)
plt.xlabel('洪水概率')
plt.ylabel('累计概率')
plt.title('洪水概率的累积分布函数图')
plt.show()
