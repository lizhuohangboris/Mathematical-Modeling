# 定义每种情境下的成本和损失数据
situations = {
    1: {"零配件1次品率": 0.10, "零配件1检测成本": 2, "零配件2次品率": 0.10, "零配件2检测成本": 3,
        "装配成本": 6, "成品次品率": 0.10, "成品检测成本": 3, "市场售价": 56, "调换损失": 6, "拆解费用": 5},
    2: {"零配件1次品率": 0.20, "零配件1检测成本": 2, "零配件2次品率": 0.20, "零配件2检测成本": 3,
        "装配成本": 6, "成品次品率": 0.20, "成品检测成本": 3, "市场售价": 56, "调换损失": 6, "拆解费用": 5},
    3: {"零配件1次品率": 0.10, "零配件1检测成本": 2, "零配件2次品率": 0.10, "零配件2检测成本": 3,
        "装配成本": 6, "成品次品率": 0.10, "成品检测成本": 3, "市场售价": 56, "调换损失": 30, "拆解费用": 5},
    4: {"零配件1次品率": 0.20, "零配件1检测成本": 1, "零配件2次品率": 0.20, "零配件2检测成本": 1,
        "装配成本": 6, "成品次品率": 0.20, "成品检测成本": 2, "市场售价": 56, "调换损失": 30, "拆解费用": 5},
    5: {"零配件1次品率": 0.10, "零配件1检测成本": 8, "零配件2次品率": 0.20, "零配件2检测成本": 1,
        "装配成本": 6, "成品次品率": 0.10, "成品检测成本": 2, "市场售价": 56, "调换损失": 10, "拆解费用": 5},
    6: {"零配件1次品率": 0.05, "零配件1检测成本": 2, "零配件2次品率": 0.05, "零配件2检测成本": 3,
        "装配成本": 6, "成品次品率": 0.05, "成品检测成本": 3, "市场售价": 56, "调换损失": 10, "拆解费用": 40}
}

# 定义递归动态规划函数
def min_cost(situation, i, j, k):
    # 获取当前情境下的数据
    data = situations[situation]
    
    # 如果检测零配件1，计算检测成本和可能的次品损失
    if i == 1:
        cost_i = data["零配件1检测成本"]
    else:
        cost_i = data["零配件1次品率"] * data["市场售价"]

    # 如果检测零配件2，计算检测成本和可能的次品损失
    if j == 1:
        cost_j = data["零配件2检测成本"]
    else:
        cost_j = data["零配件2次品率"] * data["市场售价"]

    # 如果检测成品，计算检测成本和可能的拆解或调换成本
    if k == 1:
        cost_k = data["成品检测成本"] + min(data["拆解费用"], data["调换损失"])
    else:
        cost_k = data["调换损失"]

    # 总成本计算
    total_cost = cost_i + cost_j + cost_k + data["装配成本"]
    return total_cost

# 测试每个情境的最优决策组合
for situation in range(1, 7):
    # 遍历所有可能的检测和不检测组合
    result = min_cost(situation, 1, 1, 1)  # 假设所有阶段都检测
    print(f"情境 {situation} 的最优总成本为: {result}")
