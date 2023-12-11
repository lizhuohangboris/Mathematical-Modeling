import numpy as np
import pandas as pd
from openpyxl import Workbook

# 已知参数
D = 70  # 海水深度（米）
a_deg = 1.5  # 坡度（度）
opening_angle_deg = 120  # 多波束换能器的开角（度）
distances_to_center = np.array([-800, -600, -400, -200, 0, 200, 400, 600, 800])  # 测线距中心点的距离（米）

# 将角度转换为弧度
a_rad = np.deg2rad(a_deg)
opening_angle_rad = np.deg2rad(opening_angle_deg)

# 计算覆盖宽度（W）
W = 2 * D * np.tan(a_rad)

# 初始化重叠率列表
overlap_rates = []

# 计算相邻条带之间的重叠率
for i in range(len(distances_to_center) - 1):
    d = distances_to_center[i + 1] - distances_to_center[i]
    R = 1 - (d / W)
    overlap_rates.append(R * 100)  # 将重叠率转换为百分比并添加到列表中

# 创建DataFrame保存结果
data = {
    "测线距中心点处的距离/m": distances_to_center,
    "海水深度/m": [D] * len(distances_to_center),
    "覆盖宽度/m": [W] * len(distances_to_center),
    "与前一条测线的重叠率/%": [None] + overlap_rates,
}

df = pd.DataFrame(data)

# 创建Excel文件并保存结果
wb = Workbook()
ws = wb.active
ws.title = "问题1计算结果"

# 将DataFrame写入Excel
for r_idx, row in enumerate(df.iterrows(), start=1):
    for c_idx, value in enumerate(row[1], start=1):
        ws.cell(row=r_idx, column=c_idx, value=value)

# 保存Excel文件
wb.save("result1.xlsx")

# 打印结果
print("计算结果已保存到result1.xlsx文件。")