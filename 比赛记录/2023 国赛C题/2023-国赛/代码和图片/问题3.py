import numpy as np

# 给定参数
L_south_north = 2  # 南北长度（海里）
L_east_west = 4  # 东西宽度（海里）
D = 110  # 海水深度（米）
a_degrees = 120  # 多波束换能器的开角（度）
slope_degrees = 1.5  # 坡度（度）

# 将角度转换为弧度
a_rad = np.deg2rad(a_degrees)
slope_rad = np.deg2rad(slope_degrees)

# 计算每条测线的覆盖宽度
W = 2 * D * np.tan(a_rad) * np.cos(slope_rad)

# 计算重叠率的范围
overlap_min = 0.10  # 最小重叠率（10%）
overlap_max = 0.20  # 最大重叠率（20%）

# 计算每条测线的长度
L_line = L_south_north / np.cos(slope_rad)

# 计算所需的测线数目
N = int(np.ceil(L_east_west / (W * (1 - overlap_max))))

# 计算总测线长度
L_total = N * L_line

# 打印结果
print(f"总测线长度（海里）：{L_total}")
print(f"测线数目：{N}")
print(f"每条测线长度（海里）：{L_line}")
print(f"覆盖宽度（米）：{W}")
print(f"重叠率范围：{overlap_min * 100}% 到 {overlap_max * 100}%")
