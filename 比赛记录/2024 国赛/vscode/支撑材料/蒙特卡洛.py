import numpy as np
import pandas as pd

# 六种情景的参数设置
scenarios = [
    {"P1": 0.1, "C1": 4, "C_det1": 2, "P2": 0.1, "C2": 18, "C_det2": 3, "Pf": 0.1, "C_assemble": 6, "C_detf": 3, "S": 56, "L": 6, "C_recycle": 5},
    {"P1": 0.2, "C1": 4, "C_det1": 2, "P2": 0.2, "C2": 18, "C_det2": 3, "Pf": 0.2, "C_assemble": 6, "C_detf": 3, "S": 56, "L": 6, "C_recycle": 5},
    {"P1": 0.1, "C1": 4, "C_det1": 2, "P2": 0.1, "C2": 18, "C_det2": 3, "Pf": 0.1, "C_assemble": 6, "C_detf": 3, "S": 56, "L": 30, "C_recycle": 5},
    {"P1": 0.2, "C1": 4, "C_det1": 1, "P2": 0.2, "C2": 18, "C_det2": 1, "Pf": 0.2, "C_assemble": 6, "C_detf": 2, "S": 56, "L": 30, "C_recycle": 5},
    {"P1": 0.1, "C1": 4, "C_det1": 8, "P2": 0.2, "C2": 18, "C_det2": 1, "Pf": 0.1, "C_assemble": 6, "C_detf": 2, "S": 56, "L": 10, "C_recycle": 5},
    {"P1": 0.05, "C1": 4, "C_det1": 2, "P2": 0.05, "C2": 18, "C_det2": 3, "Pf": 0.05, "C_assemble": 6, "C_detf": 3, "S": 56, "L": 10, "C_recycle": 40},
]

# 决策方案变量 (1 表示检测，0 表示不检测)
A1_options = [0, 1]  # 是否检测零配件1
A2_options = [0, 1]  # 是否检测零配件2
B_options = [0, 1]   # 是否检测成品
C_options = [0, 1]   # 是否拆解不合格成品

# 模拟参数
n_simulations = 3000  # 蒙特卡洛模拟次数

# 保存每个场景的结果
all_scenario_results = []

# 遍历六个情景
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
                        purchase_cost = params["C1"] + params["C2"]

                        # 零配件1检测决定
                        if A1:
                            while True:
                                part1_good = np.random.rand() >= params["P1"]
                                purchase_cost += params["C_det1"]  # 检测零配件1
                                if part1_good:
                                    break
                                else:
                                    purchase_cost += params["C1"]  # 如果零配件1不合格，需要重新购买
                        else:
                            part1_good = np.random.rand() >= params["P1"]  # 零配件1是否合格

                        # 零配件2检测决定
                        if A2:
                            while True:
                                part2_good = np.random.rand() >= params["P2"]
                                purchase_cost += params["C_det2"]  # 检测零配件2
                                if part2_good:
                                    break
                                else:
                                    purchase_cost += params["C2"]  # 如果零配件2不合格，需要重新购买
                        else:
                            part2_good = np.random.rand() >= params["P2"]  # 零配件2是否合格

                        # 装配阶段
                        if part1_good and part2_good:
                            product_good = np.random.rand() >= params["Pf"]  # 装配成品的合格率
                        else:
                            product_good = False  # 只要有一个零配件不合格，成品就不合格

                        
                        # 成品检测决定
                        if B:
                            purchase_cost += params["C_detf"]  # 成品检测成本
                            if not product_good:
                                product_market = False  # 成品不合格
                                # 拆解决定
                                if C:
                                    while 1:
                                        purchase_cost += params["C_recycle"]  # 拆解费用
                                        
                                        if A1 != 1:
                                            purchase_cost += params["C_det1"] 
                                        if A2 != 1:
                                            purchase_cost += params["C_det2"]
                                        
                                        if part1_good == False :
                                            while True:
                                                purchase_cost += params["C1"]  # 如果零配件1不合格，需要重新购买
                                                part1_good = np.random.rand() >= params["P1"]
                                                purchase_cost += params["C_det1"]  # 检测零配件1
                                                if part1_good:
                                                    break
                                                    
                                        if part2_good == False:
                                            while True:
                                                purchase_cost += params["C2"]  # 如果零配件1不合格，需要重新购买
                                                part2_good = np.random.rand() >= params["P2"]
                                                purchase_cost += params["C_det2"]  # 检测零配件1300
                                                if part2_good:
                                                    break
                                                    
                                        if part1_good and part2_good:
                                            product_good = np.random.rand() >= params["Pf"]
                                        else:
                                            product_good = False
                                            
                                        if product_good:
                                            total_profit += params["S"] - purchase_cost - params["C_assemble"]  # 成品进入市场的收益
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
                            # 拆解决定
                            if C:
                                while 1:
                                    purchase_cost += params["C_recycle"]  # 拆解费用
                                    
                                    if A1 != 1:
                                        purchase_cost += params["C_det1"] 
                                    if A2 != 1:
                                        purchase_cost += params["C_det2"]
                                    
                                    if part1_good == False :
                                        while True:
                                            purchase_cost += params["C1"]  # 如果零配件1不合格，需要重新购买
                                            part1_good = np.random.rand() >= params["P1"]
                                            purchase_cost += params["C_det1"]  # 检测零配件1
                                            if part1_good:
                                                break
                                                
                                    if part2_good == False:
                                        while True:
                                            purchase_cost += params["C2"]  # 如果零配件1不合格，需要重新购买
                                            part2_good = np.random.rand() >= params["P2"]
                                            purchase_cost += params["C_det2"]  # 检测零配件1300
                                            if part2_good:
                                                break
                                                
                                    if part1_good and part2_good:
                                        product_good = np.random.rand() >= params["Pf"]
                                    else:
                                        product_good = False
                                        
                                    if product_good:
                                        total_profit += params["S"] - purchase_cost - params["C_assemble"]  # 成品进入市场的收益
                                        break
                                    else:
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

# 显示每个情景的最优决策前几条
all_scenario_results