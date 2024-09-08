import numpy as np
import pandas as pd

def simulate_scenarios(scenarios, A1_options, A2_options, B_options, C_options, n_simulations):
    all_scenario_results = []

    # 遍历情景
    for scenario_index, params in enumerate(scenarios):
        results = []

        # 蒙特卡洛模拟
        for A1 in A1_options:
            for A2 in A2_options:
                for B in B_options:
                    for C in C_options:
                        total_profit = 0

                        for _ in range(n_simulations):
                            # 购买成本
                            purchase_cost = params["C1"] + params["C2"] + params["C3"] + params["C4"] + \
                                            params["C5"] + params["C6"] + params["C7"] + params["C8"]

                            # 零配件检测决定
                            for i, (A, part_key) in enumerate(zip([A1, A2], ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"])):
                                if A:
                                    while True:
                                        part_good = np.random.rand() >= params[part_key]
                                        purchase_cost += params["C_det" + str(i + 1)]  # 检测零配件
                                        if part_good:
                                            break
                                        else:
                                            purchase_cost += params["C" + str(i + 1)]  # 如果零配件不合格，需要重新购买
                                else:
                                    part_good = np.random.rand() >= params[part_key]  # 零配件是否合格

                            # 装配阶段，假设所有零配件合格才有可能成品合格
                            product_good = np.random.rand() >= params["Pf"]

                            # 成品检测决定
                            if B:
                                purchase_cost += params["C_detf"]  # 成品检测成本
                                if not product_good:
                                    product_market = False  # 成品不合格
                                else:
                                    product_market = True
                            else:
                                product_market = product_good  # 不检测直接进入市场

                            # 如果成品不合格
                            if not product_market:
                                # 拆解决定
                                if C:
                                    purchase_cost += params["C_recycle"]  # 拆解费用
                                    total_profit -= purchase_cost + params["L"]  # 不合格的成品损失
                                else:
                                    total_profit -= purchase_cost + params["L"]  # 不拆解时的损失
                            else:
                                total_profit += params["S"] - purchase_cost - params["C_assemble"]  # 成品进入市场的收益

                        # 计算平均收益
                        avg_profit = total_profit / n_simulations
                        results.append([A1, A2, B, C, avg_profit])

        # 将结果保存为DataFrame
        df_results = pd.DataFrame(results, columns=['检测零配件1', '检测零配件2', '检测成品', '拆解成品', '平均收益'])

        # 按照平均收益从高到低排序
        df_sorted_results = df_results.sort_values(by='平均收益', ascending=False)

        # 保存每个情景的结果
        all_scenario_results.append(df_sorted_results.head())

    return all_scenario_results

# 参数设置
scenarios = [
    {"P1": 0.1, "C1": 2, "C_det1": 4, "P2": 0.1, "C2": 8, "C_det2": 4, "P3": 0.1, "C3": 12, "C_det3": 4,
     "P4": 0.1, "C4": 2, "C_det4": 4, "P5": 0.1, "C5": 8, "C_det5": 4, "P6": 0.1, "C6": 12, "C_det6": 4,
     "P7": 0.1, "C7": 12, "C_det7": 4, "P8": 0.1, "C8": 12, "C_det8": 4,
     "Pf": 0.1, "C_assemble": 8, "C_detf": 6, "S": 200, "L": 40, "C_recycle": 10},
]

A1_options = [0, 1]  # 是否检测零配件1
A2_options = [0, 1]  # 是否检测零配件2
B_options = [0, 1]   # 是否检测成品
C_options = [0, 1]   # 是否拆解不合格成品
n_simulations = 500  # 蒙特卡洛模拟次数

# 调用函数
all_scenario_results = simulate_scenarios(scenarios, A1_options, A2_options, B_options, C_options, n_simulations)

# 显示结果
for idx, df in enumerate(all_scenario_results):
    print(f"Scenario {idx+1} results:")
    print(df)
