import numpy as np

# 动态决策函数，用于计算每种情况下的最优决策
def dynamic_decision(n, m, p_parts, c_detection_parts, c_assembly, c_product_detection, c_market, c_exchange, c_dis):
    # 状态空间初始化
    # 2^n 种零配件检测组合，2^m 种半成品检测组合，2 种成品检测组合，2 种拆解组合
    states = []
    for x in range(2**n):  # 零配件检测组合
        for y in range(2**m):  # 半成品检测组合
            for z in [0, 1]:  # 成品检测
                for w in [0, 1]:  # 拆解不合格成品
                    states.append((x, y, z, w))
    
    # 初始化DP表
    dp = {}
    decision = {}

    # 边界条件：初始状态时，不做任何检测和拆解
    dp[(0, 0, 0, 0)] = 0

    # 成本计算函数
    def cost(x, y, z, w):
        # 计算零配件的总检测成本
        total_c_detection_parts = sum(c_detection_parts[i] * ((x >> i) & 1) for i in range(n))
        
        # 计算未检测零配件带来的次品率（即次品率传递到后续工序）
        total_p_parts = sum(p_parts[i] * (1 - ((x >> i) & 1)) for i in range(n))
        
        # 计算半成品的装配成本（根据是否检测半成品）
        total_c_assembly = c_assembly * bin(y).count('1')
        
        # 成品的次品率由零配件次品率决定
        product_defect_rate = total_p_parts / n
        
        # 成品的检测成本
        total_c_product_detection = c_product_detection * z
        
        # 成品的总成本（检测或未检测成品的调换损失）
        total_cost_product = (1 - z) * product_defect_rate * (c_market + c_exchange)
        
        # 拆解不合格成品的成本
        total_c_dis = w * c_dis
        total_cost_disassemble = (1 - w) * product_defect_rate * (c_market + c_exchange)
        
        return total_c_detection_parts + total_c_assembly + total_c_product_detection + total_cost_product + total_cost_disassemble

    # 动态规划求解
    for state in states:
        x, y, z, w = state
        dp[state] = cost(x, y, z, w)
        decision[state] = (x, y, z, w)
    
    # 寻找最优决策
    optimal_state = min(dp, key=dp.get)
    optimal_cost = dp[optimal_state]
    
    return optimal_state, optimal_cost

# 示例参数设置（根据实际情况调整）
n = 2  # 例：2 个零配件
m = 1  # 例：1 道工序
p_parts = [0.10, 0.15]  # 零配件的次品率
c_detection_parts = [2, 3]  # 零配件的检测成本
c_assembly = 6  # 每道工序的装配成本
c_product_detection = 3  # 成品的检测成本
c_market = 56  # 成品的市场售价
c_exchange = 6  # 调换损失
c_dis = 5  # 拆解费用

# 计算最优决策
optimal_state, optimal_cost = dynamic_decision(n, m, p_parts, c_detection_parts, c_assembly, c_product_detection, c_market, c_exchange, c_dis)
print(f"最优决策: 零配件检测={bin(optimal_state[0])[2:].zfill(n)}, 半成品检测={bin(optimal_state[1])[2:].zfill(m)}, 成品检测={optimal_state[2]}, 拆解={optimal_state[3]}")
print(f"最小成本: {optimal_cost} 元")
