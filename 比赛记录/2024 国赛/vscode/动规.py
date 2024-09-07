def calculate_max_profit_advanced(cases):
    results = []
    
    for case in cases:
        defect_rate1, cost_part1, cost_test1, defect_rate2, cost_part2, cost_test2, defect_rate_final, cost_assembly, cost_test_final, sale_price, cost_exchange, cost_disassembly = case

        max_profit = -float('inf')
        best_strategy = None

        # 遍历所有可能的决策组合
        for decision1 in range(2):  # 零配件1检测决策
            for decision2 in range(2):  # 零配件2检测决策
                for decision3 in range(2):  # 成品检测决策
                    for decision4 in range(2):  # 不合格成品处理决策
                        # 计算成本和收益
                        if decision1 == 1:
                            cost1 = cost_part1 + cost_test1
                            effective_defect_rate1 = 0
                        else:
                            cost1 = cost_part1
                            effective_defect_rate1 = defect_rate1

                        if decision2 == 1:
                            cost2 = cost_part2 + cost_test2
                            effective_defect_rate2 = 0
                        else:
                            cost2 = cost_part2
                            effective_defect_rate2 = defect_rate2

                        # 成品合格率
                        effective_defect_rate_final = max(effective_defect_rate1, effective_defect_rate2, defect_rate_final)

                        if decision3 == 1:
                            assembly_cost = cost_assembly + cost_test_final
                            final_defect_rate = 0
                        else:
                            assembly_cost = cost_assembly
                            final_defect_rate = effective_defect_rate_final

                        # 成品销售收益
                        revenue = sale_price * (1 - final_defect_rate)

                        # 不合格成品处理成本
                        if decision4 == 1:
                            return_cost = cost_disassembly + (cost1 + cost2) * final_defect_rate + cost_exchange * final_defect_rate
                        else:
                            return_cost = 0

                        # 总成本和总收益
                        total_cost = cost1 + cost2 + assembly_cost + return_cost
                        profit = revenue - total_cost

                        # 更新最大利润和最佳策略
                        if profit > max_profit:
                            max_profit = profit
                            best_strategy = (decision1, decision2, decision3, decision4)

        results.append({
            'case': case,
            'max_profit': max_profit,
            'best_strategy': best_strategy
        })

    return results

# 定义六种情况的参数
cases = [
    (0.10, 4, 2, 0.10, 18, 3, 0.10, 6, 3, 56, 6, 5),
    (0.20, 4, 2, 0.20, 18, 3, 0.20, 6, 3, 56, 6, 5),
    (0.10, 4, 2, 0.10, 18, 3, 0.10, 6, 3, 56, 30, 5),
    (0.20, 4, 1, 0.20, 18, 1, 0.20, 6, 2, 56, 30, 5),
    (0.10, 4, 8, 0.20, 18, 1, 0.10, 6, 2, 56, 10, 5),
    (0.05, 4, 2, 0.05, 18, 3, 0.05, 6, 3, 56, 10, 40)
]

# 计算每种情况的最大收益和最佳决策方案
print(calculate_max_profit_advanced(cases))
