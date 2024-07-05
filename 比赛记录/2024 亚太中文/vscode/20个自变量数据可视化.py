import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 读取CSV文件，指定编码为'gbk'
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\train.csv'
data = pd.read_csv(file_path, encoding='gbk')

# 显示前几行数据
print(data.head())

# 计算相关性矩阵
correlation_matrix = data.corr()

# 提取与洪水概率相关的相关性
flood_correlation = correlation_matrix['洪水概率'].sort_values(ascending=False)

# 显示相关性
print(flood_correlation)

# 选择前五个相关性最高的指标
top_features = ['基础设施恶化', '季风强度', '大坝质量', '地形排水', '河流管理']

# 设置绘图区域大小
plt.figure(figsize=(14, 8))

# 设置字体属性
font_properties = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14)  # 使用适当的中文字体路径

# 生成散点图
for i, feature in enumerate(top_features, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=data[feature], y=data['洪水概率'])
    plt.title(f'{feature} vs 洪水概率', fontproperties=font_properties)
    plt.xlabel(feature, fontproperties=font_properties)
    plt.ylabel('洪水概率', fontproperties=font_properties)

# 调整布局
plt.tight_layout()
plt.show()
