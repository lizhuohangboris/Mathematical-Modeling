import random

# 遗传算法的参数
POPULATION_SIZE = 50  # 种群大小
GENERATIONS = 100  # 迭代次数
MUTATION_RATE = 0.01  # 变异概率
CROSSOVER_RATE = 0.7  # 交叉概率

# 初始化种群
def initialize_population(size):
    return [random.choices([0, 1], k=13) for _ in range(size)]

# 适应度函数
def fitness_function(individual):
    return calculate_expected_profit(individual)

# 选择操作（轮盘赌选择）
def selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    pick = random.uniform(0, total_fitness)
    current = 0
    for i, fitness in enumerate(fitness_scores):
        current += fitness
        if current > pick:
            return population[i]

# 交叉操作
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2

# 变异操作
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]
    return individual

# 遗传算法主流程
def genetic_algorithm():
    population = initialize_population(POPULATION_SIZE)
    for generation in range(GENERATIONS):
        fitness_scores = [fitness_function(individual) for individual in population]
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1 = selection(population, fitness_scores)
            parent2 = selection(population, fitness_scores)
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.append(mutate(offspring1))
            new_population.append(mutate(offspring2))
        population = new_population
    # 返回最佳个体
    fitness_scores = [fitness_function(individual) for individual in population]
    best_individual = population[fitness_scores.index(max(fitness_scores))]
    return best_individual, max(fitness_scores)

# 运行遗传算法
best_solution_ga, max_profit_ga = genetic_algorithm()

print(f"遗传算法最优解: {best_solution_ga}, 最大期望利润: {max_profit_ga}")
