import geopandas as gpd
import matplotlib.pyplot as plt

# 加载天津市行政区划边界数据
tianjin_boundary = gpd.read_file('D:/Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 统计建模/vscode/Tianjin-2020.shp')  # 替换为实际的行政区划边界数据文件路径

# 加载污染物年均浓度数据
pollutant_data = gpd.read_file('D:/Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 统计建模/vscode/output.shp')  # 替换为实际的污染物年均浓度数据文件路径


# 空间连接
merged_data = gpd.sjoin(tianjin_boundary, pollutant_data, how='left', op='intersects')

# # 绘制空间分布特征
# fig, ax = plt.subplots(figsize=(10, 10))
# tianjin_boundary.plot(ax=ax, color='lightgray', edgecolor='black')
# merged_data.plot(ax=ax, column='AQI', cmap='Reds', legend=True)
# plt.title('天津市大气污染物年均浓度空间分布')
# plt.xlabel('经度')
# plt.ylabel('纬度')
# plt.show()

print(merged_data.head())