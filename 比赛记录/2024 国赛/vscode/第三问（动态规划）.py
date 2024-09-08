# 定义零配件、半成品、成品的参数
n_parts = 8  # 零配件数量
n_subproducts = 3  # 半成品数量
parts_data = [ # 零配件数据，包括次品率，检测成本，拆解成本等
    {"次品率": 0.1, "检测成本": 2, "拆解成本": 6},
    # 其他零配件的数据...
]
subproducts_data = [ # 半成品数据，包括次品率，检测成本，拆解成本等
    {"次品率": 0.1, "检测成本": 4, "拆解成本": 10},
    # 其他半成品的数据...
]
product_data = {"次品率": 0.08, "市场价格": 200, "组装成本": 12, "检测成本": 8, "调换损失": 20}

# 初始化DP表格
dp = [[[0] * (n_parts + 1) for _ in range(n_subproducts + 1)] for _ in range(2)]  # DP表，考虑检测与不检测两种情况

# 填充DP表格
for part in range(n_parts):
    dp[part][检测] = min(parts_data[part]["检测成本"] + 继续到下一阶段, parts_data[part]["不检测成本"] + 继续到下一阶段)

for subproduct in range(n_subproducts):
    dp[subproduct][检测] = min(subproducts_data[subproduct]["检测成本"] + 继续到下一阶段, subproducts_data[subproduct]["不检测成本"] + 拆解成本)

dp[成品] = max(dp[成品][检测] - 总成本, dp[成品][不检测] - 总成本 - 调换损失)

# 最终输出
print("最优检测决策方案:", dp)
