import itertools

# 定义输入数据
component_defect_rates = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]  # 零配件的次品率
component_purchase_costs = [2, 8, 12, 2, 8, 12, 8, 12]  # 零配件购买成本
component_detection_costs = [1, 1, 2, 1, 1, 2, 1, 2]  # 零配件检测成本

semi_product_defect_rates = [0.10, 0.10, 0.10]  # 半成品的次品率
semi_product_assembly_costs = [8, 8, 8]  # 半成品装配成本
semi_product_detection_costs = [4, 4, 4]  # 半成品的检测成本

product_defect_rate = 0.10  # 成品次品率
product_assembly_cost = 8  # 成品装配成本
product_detection_cost = 10  # 成品检测成本
product_market_value = 200  # 成品市场售价
disassembly_cost = 40  # 拆解成本
salvage_value = 200  # 拆解后的回收价值

# 计算每种决策组合的期望利润
def calculate_expected_profit(decision):
    comp_detection = decision[:8]  # 零配件检测决策
    semi_detection = decision[8:11]  # 半成品检测决策
    prod_detection = decision[11]  # 成品检测决策
    disassemble_defective = decision[12]  # 拆解次品决策
    
    # 计算零配件阶段的成本
    total_component_cost = sum(component_purchase_costs)  # 所有零配件购买成本
    total_component_detection_cost = sum([component_detection_costs[i] for i in range(8) if comp_detection[i]])  # 零配件检测成本

    # 计算半成品阶段的成本
    total_semi_product_cost = sum(semi_product_assembly_costs)  # 半成品装配成本
    total_semi_product_detection_cost = sum([semi_product_detection_costs[i] for i in range(3) if semi_detection[i]])  # 半成品检测成本

    # 计算成品阶段的成本
    total_product_cost = product_assembly_cost  # 成品装配成本
    if prod_detection:
        total_product_cost += product_detection_cost  # 成品检测成本

    # 假设次品成品会被拆解，计算拆解费用
    total_disassembly_cost = disassembly_cost if disassemble_defective else 0
    
    # 总成本
    total_cost = (total_component_cost + total_component_detection_cost +
                  total_semi_product_cost + total_semi_product_detection_cost +
                  total_product_cost + total_disassembly_cost)
    
    # 成品的期望利润计算
    product_prob_defective = product_defect_rate  # 假设成品次品率为固定值 10%
    expected_profit = (1 - product_prob_defective) * product_market_value + product_prob_defective * salvage_value - total_cost
    
    return expected_profit

# 枚举所有可能的决策组合（2^13 种组合）
def find_optimal_solution():
    decisions = list(itertools.product([0, 1], repeat=13))  # 枚举所有可能的决策组合
    best_decision = None
    max_profit = float('-inf')

    # 遍历所有组合，计算每个组合的期望利润，找到最大值
    for decision in decisions:
        profit = calculate_expected_profit(decision)
        if profit > max_profit:
            max_profit = profit
            best_decision = decision

    return best_decision, max_profit

# 找到最优解
best_decision, max_profit = find_optimal_solution()
print(f"最优决策组合: {best_decision}")
print(f"最大期望利润: {max_profit}")
