import numpy as np
import pandas as pd

# 敏感度分析的实现
def run_simulation_with_params(scenario, A1_options, A2_options, B_options, C_options, n_simulations):
    results = []

    # 蒙特卡洛模拟
    for A1 in A1_options:
        for A2 in A2_options:
            for B in B_options:
                for C in C_options:
                    total_profit = 0

                    for _ in range(n_simulations):
                        # 购买成本
                        purchase_cost = scenario["C1"] + scenario["C2"]

                        # 零配件1检测决定
                        if A1:
                            while True:
                                part1_good = np.random.rand() >= scenario["P1"]
                                purchase_cost += scenario["C_det1"]  # 检测零配件1
                                if part1_good:
                                    break
                                else:
                                    purchase_cost += scenario["C1"]  # 如果零配件1不合格，需要重新购买
                        else:
                            part1_good = np.random.rand() >= scenario["P1"]  # 零配件1是否合格

                        # 零配件2检测决定
                        if A2:
                            while True:
                                part2_good = np.random.rand() >= scenario["P2"]
                                purchase_cost += scenario["C_det2"]  # 检测零配件2
                                if part2_good:
                                    break
                                else:
                                    purchase_cost += scenario["C2"]  # 如果零配件2不合格，需要重新购买
                        else:
                            part2_good = np.random.rand() >= scenario["P2"]  # 零配件2是否合格

                        # 装配阶段
                        if part1_good and part2_good:
                            product_good = np.random.rand() >= scenario["Pf"]  # 装配成品的合格率
                        else:
                            product_good = False  # 只要有一个零配件不合格，成品就不合格

                        # 成品检测决定
                        if B:
                            purchase_cost += scenario["C_detf"]  # 成品检测成本
                            if not product_good:
                                product_market = False  # 成品不合格
                                # 拆解决定
                                if C:
                                    while 1:
                                        purchase_cost += scenario["C_recycle"]  # 拆解费用
                                        if part1_good == False:
                                            while True:
                                                purchase_cost += scenario["C1"]  # 如果零配件1不合格，需要重新购买
                                                part1_good = np.random.rand() >= scenario["P1"]
                                                purchase_cost += scenario["C_det1"]  # 检测零配件1
                                                if part1_good:
                                                    break
                                        if part2_good == False:
                                            while True:
                                                purchase_cost += scenario["C2"]  # 如果零配件1不合格，需要重新购买
                                                part2_good = np.random.rand() >= scenario["P2"]
                                                purchase_cost += scenario["C_det2"]  # 检测零配件1300
                                                if part2_good:
                                                    break
                                        if part1_good and part2_good:
                                            product_good = np.random.rand() >= scenario["Pf"]
                                        else:
                                            product_good = False
                                        if product_good:
                                            total_profit += scenario["S"] - purchase_cost - scenario["C_assemble"]  # 成品进入市场的收益
                                            break
                                        else:
                                            total_profit -= purchase_cost  # 不合格的成品损失
                                    continue
                            else:
                                product_market = True
                        else:
                            product_market = product_good  # 不检测直接进入市场

                        # 如果成品不合格
                        if not product_market:
                            if C:
                                while 1:
                                    purchase_cost += scenario["C_recycle"]  # 拆解费用
                                    if part1_good == False:
                                        while True:
                                            purchase_cost += scenario["C1"]  # 如果零配件1不合格，需要重新购买
                                            part1_good = np.random.rand() >= scenario["P1"]
                                            purchase_cost += scenario["C_det1"]  # 检测零配件1
                                            if part1_good:
                                                break
                                    if part2_good == False:
                                        while True:
                                            purchase_cost += scenario["C2"]  # 如果零配件1不合格，需要重新购买
                                            part2_good = np.random.rand() >= scenario["P2"]
                                            purchase_cost += scenario["C_det2"]  # 检测零配件1300
                                            if part2_good:
                                                break
                                    if part1_good and part2_good:
                                        product_good = np.random.rand() >= scenario["Pf"]
                                    else:
                                        product_good = False
                                    if product_good:
                                        total_profit += scenario["S"] - purchase_cost - scenario["C_assemble"]  # 成品进入市场的收益
                                        break
                                    else:
                                        total_profit -= purchase_cost + scenario["L"]  # 不合格的成品损失
                            else:
                                total_profit -= purchase_cost + scenario["L"]  # 不拆解时的损失
                        else:
                            total_profit += scenario["S"] - purchase_cost - scenario["C_assemble"]  # 成品进入市场的收益

                    # 计算平均收益
                    avg_profit = total_profit / n_simulations
                    results.append([A1, A2, B, C, avg_profit])

    return pd.DataFrame(results, columns=['检测零配件1', '检测零配件2', '检测成品', '拆解成品', '平均收益'])

# 设置敏感度分析范围（±10%）
sensitivity_factors = [0.9, 1.0, 1.1]
sensitivity_results = []

# 选择一个场景进行敏感度分析，这里选取第一个场景
base_scenario = scenarios[0]

for param in ['P1', 'P2', 'Pf', 'C1', 'C2', 'S', 'L']:
    for factor in sensitivity_factors:
        modified_scenario = base_scenario.copy()
        modified_scenario[param] *= factor
        # 运行模拟并保存结果
        scenario_result = run_simulation_with_params(modified_scenario, A1_options, A2_options, B_options, C_options, n_simulations)
        sensitivity_results.append((param, factor, scenario_result['平均收益'].mean()))

# 将敏感度分析结果保存为DataFrame
sensitivity_df = pd.DataFrame(sensitivity_results, columns=['参数', '变化因子', '平均收益'])

import ace_tools as tools; tools.display_dataframe_to_user(name="Sensitivity Analysis Results", dataframe=sensitivity_df)
