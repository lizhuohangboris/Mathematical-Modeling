import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings

# 设置字体为宋体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 设置表格字体大小
# plt.rcParams['font.size'] = 5 # 进一步减小标签字体大小为5
# 使用绝对文件路径读取Excel文件
df_1_data1 = pd.read_excel(r"C:\Users\92579\Desktop\final\蔬菜品类时序图.xlsx")
# 绘制蔬菜各品类时序图（销售月份曲线-规律）
plt.figure(figsize=(28, 12), dpi=300) # 进一步增加图表宽度
plt.subplots_adjust(left=0.03, right=0.97, wspace=0.3, hspace=0.5) # 增加左右间距和上下两行的间距
n = -1
line_styles = ['-', '--', '-.', ':'] # 不同的线条样式
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] # 不同的颜色
for i, column in enumerate(df_1_data1.columns):
   s = df_1_data1[column]
   n += 1
plt.subplot(2, 4, n + 1)
sns.lineplot(data=s, linestyle=line_styles[i % len(line_styles)], color=colors[i % len(colors)])
plt.title(column, fontsize=6) # 进一步减小标签字体大小
plt.grid()
plt.xticks(rotation=45) # 旋转x轴标签
plt.savefig('蔬菜各品类销售-月份.png') # 保存图片
plt.show()