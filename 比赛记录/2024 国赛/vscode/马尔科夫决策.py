import numpy as np

# 定义次品率和检测成本
defect_rate = 0.1  # 假设零件和半成品的次品率为10%
detection_cost = 5  # 检测成本
reward_if_passed = 100  # 如果成品合格，市场价值为100
penalty_if_failed = -50  # 如果成品次品，惩罚为-50

# 定义状态空间：包括8个零件，3个半成品，1个成品的状态（0: 次品, 1: 合格）
states = [(d1, q1, d2, q2, d3, q3, d4, q4, d5, q5, d6, q6, d7, q7, d8, q8,
           ds1, qs1, ds2, qs2, ds3, qs3, dp, qp)
          for d1 in [0, 1] for q1 in [0, 1]
          for d2 in [0, 1] for q2 in [0, 1]
          for d3 in [0, 1] for q3 in [0, 1]
          for d4 in [0, 1] for q4 in [0, 1]
          for d5 in [0, 1] for q5 in [0, 1]
          for d6 in [0, 1] for q6 in [0, 1]
          for d7 in [0, 1] for q7 in [0, 1]
          for d8 in [0, 1] for q8 in [0, 1]
          for ds1 in [0, 1] for qs1 in [0, 1]
          for ds2 in [0, 1] for qs2 in [0, 1]
          for ds3 in [0, 1] for qs3 in [0, 1]
          for dp in [0, 1] for qp in [0, 1]]  # 生成所有可能的状态组合

# 定义动作空间：是否检测和是否拆解（0: 不做, 1: 做）
actions = [
    (a1, a2, a3, a4, a5, a6, a7, a8, as1, as2, as3, t1, t2, t3, ap, tp)
    for a1 in [0, 1] for a2 in [0, 1] for a3 in [0, 1] for a4 in [0, 1]
    for a5 in [0, 1] for a6 in [0, 1] for a7 in [0, 1] for a8 in [0, 1]
    for as1 in [0, 1] for as2 in [0, 1] for as3 in [0, 1]
    for t1 in [0, 1] for t2 in [0, 1] for t3 in [0, 1]
    for ap in [0, 1] for tp in [0, 1]
]

# 初始化所有状态的价值
V = {s: 0 for s in states}

# 状态转移函数
def transition_probability(state, action):
    """给定一个状态和动作，生成下一个状态的转移概率"""
    next_states = {}
    next_state = list(state)
    for i in range(0, 16, 2):  # 每两个元素表示一个零件的检测和合格状态
        if action[i//2] == 1:  # 如果选择了检测
            prob = 1 - defect_rate  # 检测后合格的概率
            next_state[i+1] = int(np.random.rand() < prob)  # 根据次品率更新零件状态
        else:
            next_state[i+1] = state[i+1]  # 如果不检测，保持当前合格状态不变
    next_states[tuple(next_state)] = 1.0  # 假设转移后一定是某个状态
    return next_states

# 奖励函数：给定一个状态和动作，计算即时奖励
def reward(state, action):
    r = 0
    for i, a in enumerate(action):
        if a == 1:  # 如果选择了检测或拆解
            if i < 8:  # 检测零件
                r -= detection_cost  # 检测的成本
            else:  # 拆解或者检测半成品和成品
                if state[-2] == 1:  # 如果成品合格
                    r += reward_if_passed
                else:
                    r += penalty_if_failed
        else:
            r -= 100  # 假设未检测出次品的惩罚为 -100
    return r

# 值迭代算法
def value_iteration(states, actions, gamma=0.9, theta=1e-6):
    while True:
        delta = 0
        for s in states:
            v = V[s]
            # 更新状态的价值：对每个动作计算期望价值
            V[s] = max(
                sum(transition_probability(s, a)[s_next] * (reward(s, a) + gamma * V[s_next])
                    for s_next in transition_probability(s, a) if s_next in V)  # 确保下一个状态在 V 中
                for a in actions
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    # 根据状态的价值选择最优策略
    policy = {}
    for s in states:
        policy[s] = max(actions, key=lambda a: sum(
            transition_probability(s, a)[s_next] * (reward(s, a) + gamma * V[s_next])
            for s_next in transition_probability(s, a) if s_next in V))  # 确保下一个状态在 V 中
    return policy, V

# 运行值迭代算法
policy, V = value_iteration(states, actions)

# 输出结果
print("Optimal policy:", policy)
print("State values:", V)
