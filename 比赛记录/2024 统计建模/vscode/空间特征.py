import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)

# 加载天津市行政区划边界数据
tianjin_boundary = gpd.read_file('D:/Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 统计建模/vscode/Tianjin-2020.shp')  # 替换为实际的行政区划边界数据文件路径

# 加载污染物年均浓度数据
pollutant_data = gpd.read_file('D:/Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 统计建模/vscode/output.shp')  # 替换为实际的污染物年均浓度数据文件路径

# 空间连接
merged_data = gpd.sjoin(tianjin_boundary, pollutant_data, how='left', op='intersects')

# 绘制空间分布特征
fig, ax = plt.subplots(figsize=(10, 10))
tianjin_boundary.plot(ax=ax, color='lightgray', edgecolor='black')
merged_data.plot(ax=ax, column='index_right', cmap='Reds', legend=True)
plt.title('天津市大气污染物年均浓度空间分布', fontproperties=font)
plt.xlabel('经度', fontproperties=font)
plt.ylabel('纬度', fontproperties=font)
# 添加观测点
obs_points = merged_data[['index_right', 'pollutant_1']].drop_duplicates()
obs_points.plot(ax=ax, x='index_right', y='pollutant_1', c='red', marker='o', alpha=0.5, label='Observation Points')

# 添加颜色变化
colors = plt.cm.Reds(np.linspace(0, 1, len(obs_points)))
for i, color in enumerate(colors):
    ax.axvline(x=i, color=color, linestyle='--', alpha=0.8)

# 设置图例
ax.legend()

# 显示图表
plt.show()
