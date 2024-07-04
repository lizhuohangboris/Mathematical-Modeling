import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024亚太中文\train.csv'
data = pd.read_csv(file_path, encoding='gbk')

# 检查数据完整性
print(data.info())

# 处理缺失值和异常值（示例）
data = data.dropna()  # 简单示例，实际可能需要更复杂的处理

# 假设 "flood_occurrence" 是洪水发生的指标列名
flood_occurrence = data['洪水概率']

# 计算各指标与洪水发生的皮尔逊相关系数
correlation_results = {}
variables = ['季风强度', '地形排水', '河流管理', '森林砍伐', '城市化', '气候变化', 
             '大坝质量', '淤积', '农业实践', '侵蚀', '无效防灾', '排水系统', '海岸脆弱性', 
             '滑坡', '流域', '基础设施恶化', '人口得分', '湿地损失', '规划不足', '政策因素']

for var in variables:
    correlation_results[var] = data[var].corr(flood_occurrence)

# 转换为 DataFrame 方便后续操作
correlation_df = pd.DataFrame(list(correlation_results.items()), columns=['Indicator', 'Pearson_Correlation_Coefficient'])

# 按相关系数排序
correlation_df = correlation_df.sort_values(by='Pearson_Correlation_Coefficient', ascending=False)

# 绘制皮尔逊相关系数柱状图
plt.figure(figsize=(12, 8))
plt.bar(correlation_df['Indicator'], correlation_df['Pearson_Correlation_Coefficient'])
plt.xticks(rotation=90)
plt.xlabel('Indicators')
plt.ylabel('Pearson Correlation Coefficient')
plt.title('Pearson Correlation Coefficient of Indicators with Flood Occurrence')
plt.show()

# 高关联度指标（相关系数大于0.5）
high_correlation_indicators = correlation_df[correlation_df['Pearson_Correlation_Coefficient'] > 0.5]

# 低关联度指标（相关系数小于0.5）
low_correlation_indicators = correlation_df[correlation_df['Pearson_Correlation_Coefficient'] < 0.5]

print("高关联度指标:")
print(high_correlation_indicators)

print("低关联度指标:")
print(low_correlation_indicators)
