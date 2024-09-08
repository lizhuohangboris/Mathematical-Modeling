import numpy as np
from scipy.stats import beta  # 导入 Beta 分布
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体，使用系统中的中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 例如使用 SimHei 字体
rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题
# 定义模型参数
num_states = 10  # 假设有10个状态
num_actions = 3  # 假设有3个动作：检测、不检测、拆解
gamma = 0.95     # 折扣因子
epsilon = 0.01   # 收敛阈值

# 定义转移概率矩阵 P 和奖励矩阵 R
P = np.zeros((num_actions, num_states, num_states))
R = np.zeros((num_states, num_actions))

# 功效值 (检测的准确性)
power_reject = 0.150  # 对应零配件和半成品缺陷率
power_accept = 0.050  # 对应成品缺陷率

# 定义组件的检测信息
components = {
    "零件1": {"defect_rate": 0.10, "detection_cost": 2},
    "零件2": {"defect_rate": 0.10, "detection_cost": 8},
    "零件3": {"defect_rate": 0.10, "detection_cost": 12},
    "零件4": {"defect_rate": 0.10, "detection_cost": 2},
    "零件5": {"defect_rate": 0.10, "detection_cost": 8},
    "零件6": {"defect_rate": 0.10, "detection_cost": 12},
}

# 贝叶斯更新函数
def bayesian_update(prior, success):
    alpha, beta_param = prior.args
    alpha += 1 if success else 0
    beta_param += 1 if not success else 0
    return beta(alpha, beta_param)

# 行为策略：为每个状态选择最优动作并计算收益
def action_selection_policy(component, transition_priors, discount_factor=0.95, epsilon=0.1):
    """
    根据当前的贝叶斯估计为每个组件选择最优动作，并考虑检测成本和长期回报
    :param component: 当前的组件（零件/半成品）
    :param transition_priors: 当前组件的转移概率分布
    :param discount_factor: 折现因子，影响长期回报的计算
    :param epsilon: 随机探索的概率（ε-greedy 策略）
    :return: 最优的动作和对应的收益
    """
    # ε-greedy 策略，引入一定的随机性探索
    if np.random.rand() < epsilon:
        return np.random.choice(['检测', '不检测', '拆解']), 0  # 随机探索时不考虑收益
    
    detection_cost = components[component]["detection_cost"]  # 检测成本
    success_reward = 10  # 假设检测成功后带来的市场潜在收益
    
    # 计算每个动作的期望收益
    expected_value = transition_priors[component][0].mean() * success_reward - detection_cost
    return '检测', expected_value

# 值迭代算法的实现
def value_iteration(P, R, gamma=0.95, epsilon=0.01):
    num_states = P.shape[1]
    num_actions = P.shape[0]
    
    V = np.zeros(num_states)  # 初始化值函数
    policy = np.zeros(num_states, dtype=int)  # 初始化策略
    
    while True:
        V_prev = np.copy(V)  # 保存上一次的值函数
        for s in range(num_states):
            Q = np.zeros(num_actions)  # 存储每个动作的 Q 值
            for a in range(num_actions):
                Q[a] = R[s, a] + gamma * sum([P[a, s, sp] * V[sp] for sp in range(num_states)])  # 计算 Q 值
            V[s] = np.max(Q)  # 更新值函数
            policy[s] = np.argmax(Q)  # 更新策略
        
        # 检查收敛条件
        if np.max(np.abs(V - V_prev)) < epsilon:
            break
    
    return policy, V

# 计算不同场景下的最优决策组合
def calculate_decision(scenarios):
    for scenario in scenarios:
        # 解包场景参数
        purchase_cost_p1, purchase_cost_p2, assembly_cost, market_price, test_cost_p1, test_cost_p2, \
        test_cost_prod, disassembly_cost, exchange_loss = scenario
        
        # 固定缺陷率值
        defect_rate_p1 = power_reject
        defect_rate_p2 = power_reject
        defect_rate_prod = power_accept

        # 遍历所有可能的决策组合
        for test_p1 in [True, False]:
            for test_p2 in [True, False]:
                for test_prod in [True, False]:
                    # 计算成本和收益
                    cost = purchase_cost_p1 + purchase_cost_p2 + assembly_cost
                    revenue = market_price

                    # 检测零配件1
                    if test_p1:
                        cost += test_cost_p1
                        if defect_rate_p1 > 0 and not test_prod:
                            cost += disassembly_cost + exchange_loss
                    
                    # 检测零配件2
                    if test_p2:
                        cost += test_cost_p2
                        if defect_rate_p2 > 0 and not test_prod:
                            cost += disassembly_cost + exchange_loss

                    # 检测成品
                    if test_prod:
                        cost += test_cost_prod
                        if defect_rate_prod > 0:
                            cost += defect_rate_prod * (disassembly_cost + exchange_loss)
                            revenue -= defect_rate_prod * exchange_loss
                    else:
                        cost += defect_rate_prod * (disassembly_cost + exchange_loss)
                        revenue -= defect_rate_prod * exchange_loss

                    # 计算净收益
                    net_cost = cost - revenue
                    print(f"检测决策组合 (p1: {test_p1}, p2: {test_p2}, prod: {test_prod}), 净成本: {net_cost}")

# 定义场景数据
scenarios = [
    # (零配件1购买成本, 零配件2购买成本, 装配成本, 市场售价, 零配件1检测成本, 零配件2检测成本, 成品检测成本, 拆解成本, 调换损失)
    (2, 8, 10, 50, 1, 2, 6, 5, 10),
    (2, 8, 10, 55, 1, 2, 6, 5, 15),
    (2, 8, 12, 54, 1, 2, 6, 6, 12),
]

# 计算最优决策
calculate_decision(scenarios)

# 假设P和R矩阵已初始化，运行值迭代算法
policy, V = value_iteration(P, R)
print("最优策略:", policy)
print("值函数:", V)

# 执行模拟 BAMDP 过程
def simulate_bamdp(num_iterations):
    current_state = 1  # 假设一开始我们在合格状态
    total_rewards = []  # 用于记录每次迭代的总收益
    transition_priors = {component: [beta(1, 1)] for component in components}  # 初始化贝叶斯先验
    
    for i in range(num_iterations):
        print(f"第 {i+1} 次迭代")
        iteration_reward = 0  # 每次迭代的总收益
        for component in components:
            print(f"当前零件: {component}, 当前状态: {current_state}")
            # 为每个零件选择动作并记录对应收益
            action, reward = action_selection_policy(component, transition_priors)
            print(f"选择动作: {action}, 预期收益: {reward}")

            # 累加该组件的预期收益
            iteration_reward += reward

            # 假设根据动作生成奖励和状态转移
            success = np.random.rand() > components[component]["defect_rate"]  # 根据次品率判断是否成功
            transition_priors[component][0] = bayesian_update(transition_priors[component][0], success)

        # 记录当前迭代的总收益
        total_rewards.append(iteration_reward)
        print(f"第 {i+1} 次迭代的总收益: {iteration_reward}\n")

    # 输出最大收益
    max_reward = max(total_rewards)
    print(f"最大收益: {max_reward}")

    # 绘制收益趋势图
    plt.plot(range(1, num_iterations + 1), total_rewards, marker='o')
    plt.title('收益趋势')
    plt.xlabel('迭代次数')
    plt.ylabel('总收益')
    plt.show()

# 执行 BAMDP 模拟
simulate_bamdp(10)
