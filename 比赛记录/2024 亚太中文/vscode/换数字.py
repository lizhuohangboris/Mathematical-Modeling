import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.font_manager as fm

# 数据
data = {
   "指标": ["季风强度", "地形排水", "河流管理", "森林砍伐", "农业实践", "无效防灾", "海岸脆弱性型", "流域", "人口得分", "规划不足",
             "城市化", "气候变化", "大坝质量", "淤积", "侵蚀", "排水系统", "滑坡", "基础设施恶化", "湿地损失", "政策因素"],
   "灰色关联度值": [0.707628, 0.681653, 0.715706, 0.688852, 0.716449, 0.700754, 0.703549, 0.714254, 0.689276, 0.714684,
                    0.692787, 0.707646, 0.715536, 0.727431, 0.69934, 0.703328, 0.706461, 0.69502, 0.675788, 0.715782]
}

df = pd.DataFrame(data)

# 加载中文字体
font_path = r'C://Windows/Fonts/simhei.ttf' # 请更换为你本地字体的路径，例如 'C:/Windows/Fonts/simhei.ttf'
prop = fm.FontProperties(fname=font_path)

# 绘制热力图
plt.figure(figsize=(22,2))
heatmap = sns.heatmap(df.set_index("指标").T, annot=True, cmap='coolwarm', cbar=True, annot_kws={"size": 12, "fontproperties": prop})
heatmap.set_title("灰色关联度矩阵热力图", fontsize=22, fontproperties=prop)
plt.xticks(rotation=45, fontsize=16, fontproperties=prop)
plt.yticks(rotation=0, fontsize=16, fontproperties=prop)


# 显示图表
plt.show()
