import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import font_manager as fm

# 设置字体路径
font_path = 'C:/Windows/Fonts/simhei.ttf'  # 替换为你的系统中黑体字体的路径
font_prop = fm.FontProperties(fname=font_path)

# 数据准备
data = {
    '国家': ['美国', '美国', '法国', '法国', '德国', '德国', 
          '美国', '美国', '法国', '法国', '德国', '德国'],
    '宠物种类': ['猫', '狗', '猫', '狗', '猫', '狗', 
             '猫', '狗', '猫', '狗', '猫', '狗'],
    '2023': [7380, 8010, 1660, 990, 1570, 1050, 
             7380, 8970, 1490, 760, 1520, 1060],
    '2022': [7380, 8970, 1490, 760, 1520, 1060, 
             9420, 8970, 1510, 750, 1670, 1030],
    '2021': [9420, 8970, 1510, 750, 1670, 1030, 
             6500, 8500, 1490, 775, 1570, 1070],
    '2020': [6500, 8500, 1490, 775, 1570, 1070, 
             9420, 8970, 1300, 740, 1470, 1010],
    '2019': [9420, 8970, 1300, 740, 1470, 1010, 
             9420, 8970, 1300, 740, 1470, 1010]
}

# 转换为DataFrame
df = pd.DataFrame(data)

# 重新格式化数据，以便绘制小提琴图
df_melted = pd.melt(df, id_vars=['国家', '宠物种类'], 
                    value_vars=['2023', '2022', '2021', '2020', '2019'], 
                    var_name='年份', value_name='数量')

# 创建小提琴图
plt.figure(figsize=(14, 8))
sns.set(style="whitegrid", palette="muted")  # 设置图表风格和调色板
sns.violinplot(x='年份', y='数量', hue='宠物种类', data=df_melted, split=True, inner="quart")

# 添加标题和标签，并使用字体属性
plt.title('不同国家宠物数量（猫与狗）的变化趋势', fontproperties=font_prop, fontsize=18)
plt.xlabel('年份', fontproperties=font_prop, fontsize=14)
plt.ylabel('数量', fontproperties=font_prop, fontsize=14)
plt.legend(title='宠物种类', prop=font_prop)

# 显示图形
plt.tight_layout()
plt.show()
