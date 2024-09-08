from random import random
import numpy as np

from 遗传算法 import fitness_function

# 粒子群优化的参数
POPULATION_SIZE = 50  # 粒子数量
ITERATIONS = 100  # 迭代次数
W = 0.5  # 惯性权重
C1 = 1.5  # 个体学习因子
C2 = 1.5  # 群体学习因子

# 初始化粒子群
def initialize_particles():
    return [random.choices([0, 1], k=13) for _ in range(POPULATION_SIZE)]

# 粒子群优化主流程
def pso():
    particles = initialize_particles()
    velocities = [np.random.uniform(-1, 1, 13) for _ in range(POPULATION_SIZE)]
    personal_best = particles[:]
    global_best = max(particles, key=fitness_function)
    
    for iteration in range(ITERATIONS):
        for i in range(POPULATION_SIZE):
            velocities[i] = (W * np.array(velocities[i]) +
                             C1 * random.random() * (np.array(personal_best[i]) - np.array(particles[i])) +
                             C2 * random.random() * (np.array(global_best) - np.array(particles[i])))
            particles[i] = np.clip(particles[i] + velocities[i], 0, 1).round().astype(int)
            
            if fitness_function(particles[i]) > fitness_function(personal_best[i]):
                personal_best[i] = particles[i]
            if fitness_function(personal_best[i]) > fitness_function(global_best):
                global_best = personal_best[i]
    
    return global_best, fitness_function(global_best)

best_solution_pso, max_profit_pso = pso()
print(f"粒子群优化最优解: {best_solution_pso}, 最大期望利润: {max_profit_pso}")
