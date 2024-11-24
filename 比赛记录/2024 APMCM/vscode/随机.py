import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 数据准备
heatmap_data_full = {
    "国家": ["美国", "美国", "美国", "美国", "美国",
           "法国", "法国", "法国", "法国", "法国",
           "德国", "德国", "德国", "德国", "德国"],
    "年份": ["2023", "2022", "2021", "2020", "2019",
           "2023", "2022", "2021", "2020", "2019",
           "2023", "2022", "2021", "2020", "2019"],
    "数量": [7380, 7380, 9420, 6500, 9420,
            1660, 1490, 1510, 1490, 1300,
            1570, 1520, 1670, 1570, 1470]
}

# 创建 DataFrame
df_heatmap = pd.DataFrame(heatmap_data_full)

# 构造热力图数据
heatmap_pivot = df_heatmap.pivot_table(index="国家", columns="年份", values="数量")

# 绘制热力图
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_pivot, annot=True, fmt="d", cmap="coolwarm", linewidths=0.5, cbar_kws={"label": "数量"})
plt.title("不同国家和年份的宠物数量分布（热力图）", fontsize=16)
plt.xlabel("年份", fontsize=12)
plt.ylabel("国家", fontsize=12)
plt.show()
