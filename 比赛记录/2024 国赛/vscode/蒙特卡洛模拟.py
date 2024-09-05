import numpy as np

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

# 蒙特卡洛模拟函数
def monte_carlo_simulation(situation, simulations=1000):
    # 获取当前情境下的数据
    data = situations[situation]
    
    # 存储每次模拟的总成本
    total_costs = []
    
    # 进行多次模拟
    for _ in range(simulations):
        # 随机生成次品数量，使用伯努利分布模拟次品率
        parts1_defective = np.random.binomial(1, data["零配件1次品率"])  # 零配件1是否次品（0或1）
        parts2_defective = np.random.binomial(1, data["零配件2次品率"])  # 零配件2是否次品（0或1）
        product_defective = np.random.binomial(1, data["成品次品率"])      # 成品是否次品（0或1）
        
        # 零配件1的成本
        if parts1_defective:
            cost_parts1 = data["零配件1次品率"] * data["市场售价"]  # 不检测时的次品损失
        else:
            cost_parts1 = data["零配件1检测成本"]  # 检测成本
        
        # 零配件2的成本
        if parts2_defective:
            cost_parts2 = data["零配件2次品率"] * data["市场售价"]
        else:
            cost_parts2 = data["零配件2检测成本"]
        
        # 成品的成本
        if product_defective:
            cost_product = min(data["拆解费用"], data["调换损失"])  # 拆解或调换的最小损失
        else:
            cost_product = data["成品检测成本"]  # 成品检测成本
        
        # 总成本 = 零配件1成本 + 零配件2成本 + 成品成本 + 装配成本
        total_cost = cost_parts1 + cost_parts2 + cost_product + data["装配成本"]
        total_costs.append(total_cost)
    
    # 返回总成本的均值和标准差
    return np.mean(total_costs), np.std(total_costs)

# 运行蒙特卡洛模拟
for situation in range(1, 7):
    mean_cost, std_cost = monte_carlo_simulation(situation)
    print(f"情境 {situation} 的平均总成本为: {mean_cost:.2f}, 标准差为: {std_cost:.2f}")
