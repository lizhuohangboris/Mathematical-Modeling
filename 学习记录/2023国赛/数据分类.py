"""
 该程序用于蔬菜水果的分类（按照时间）
 分类数据来源：进货数据
 类别：
 - 常年可供应的蔬菜和水果：这些产品几乎整年都可以获得
 判断标准：一年中销售天数大于 300 天 共计 30 个
 - 季节性蔬菜和水果：这些产品在特定的季节内生长和销售，受气候和地理条件的影响。
 判断标准：其他 共计 151 个
 - 时令蔬菜和水果：这些产品在某些特定的节日或假期季节内销售。
 判断标准：一年中销售天数小于 15 天 共计 70 个
 """

 ############ 季节性蔬菜和水果 统计 ##########
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = [u'simHei']
plt.rcParams['axes.unicode_minus'] = False


csv_file = 'C:/Users/92579/Documents/GitHub/Mathematical-Modeling/学习记录/2023国赛/C题/附件 1.csv'
df_1 = pd.read_csv(csv_file)

csv_file = 'C:/Users/92579/Documents/GitHub/Mathematical-Modeling/学习记录/2023国赛/C题/附件 3.csv'
df = pd.read_csv(csv_file)

df['日期'] = pd.to_datetime(df['日期'])
df['月份'] = df['日期'].dt.month
 # 分品类
mapping_dict = df_1.set_index('单品编码')['分类名称'].to_dict()
df['品类'] = df['单品编码'].map(mapping_dict)
print(df.head(5))

grouped = df.groupby('单品编码')
result = {}

for name, group in grouped:
    unique_months = group['月份'].unique()
    total_months = len(unique_months)
    season = []
    season_list = [0]*4
    if 3 in unique_months or 4 in unique_months or 5 in unique_months:
        season.append("春季")
        season_list[0] = 1
    if 6 in unique_months or 7 in unique_months or 8 in unique_months:
        season.append("夏季")
        season_list[1] = 1
    if 9 in unique_months or 10 in unique_months or 11 in unique_months:
        season.append("秋季")
        season_list[2] = 1
    if 12 in unique_months or 1 in unique_months or 2 in unique_months:
        season.append("冬季")
        season_list[3] = 1
    result[name] = {
        '出现的月份': unique_months,
        '总共出现的月份数': total_months,
        '出现的季节': season,
        "季节数": len(season),
        "季节列表": season_list
    }

count_all = 0
count_all_list = []
for key, value in result.items():
    if value['季节数'] == 4:
        count_all += 1
        count_all_list.append(key)
    #print(f" 单品编码 {key} 出现在以下月份: {', '.join(map(str, value['出现的月份']))}，总共出现的月份数: {value['总共出现的月份数']}, 出现在 {value['出现的季节']}")

print(count_all)
print(count_all_list)


############ 常年可供应的蔬菜和水果 时令蔬菜和水果 统计 #############


df['年份'] = df['日期'].dt.year

result = df.groupby(['单品编码', '年份']).agg({'日期': 'nunique'}).reset_index()
result.rename(columns={'日期': '天数'}, inplace=True)

#print(result)

max_days = result.groupby('单品编码')['天数'].max().reset_index()
# print(max_days)
plt.hist(max_days['天数'], bins=35, edgecolor='k') # 可自行调整 bins 参数来设置柱子数量
plt.xlabel('天数')
plt.ylabel('频数')
plt.title('天数分布直方图')
plt.show()
filtered_df = max_days[max_days['天数'] <= 15]
cnt = 0
cnt_list = []
for index, row in filtered_df.iterrows():
    cnt_list.append(row['单品编码'])
    print(f" 单品编码：{row['单品编码']}，一年最多出现{row['天数']}天")
    cnt += 1
print(cnt)
