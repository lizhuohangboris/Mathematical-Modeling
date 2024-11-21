import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
 
# 参数设置
alpha_1 = 0.8  # 全球需求自回归系数
beta_1 = 0.2  # 全球需求对外部因素的影响系数
gamma = 0.05  # 关税对出口量的负向影响
alpha_2 = 2  # 关税的衰减系数
lambda_ = 0.1  # 关税对价格的影响
beta_2 = 1.5  # 关税的正向影响（非线性效应）
mu = 0.03  # 全球需求对出口量的影响系数
rho = 50  # 基础出口价格系数
phi = 0.05  # 全球需求对价格的弹性系数
delta = 0.02  # 全球需求对价格的影响
tau = 0.01  # 其他国家价格的影响系数
sigma_T = 0.02  # 关税波动的标准差
sigma_D = 0.05  # 全球需求波动的标准差
 
 
years = 5
cat_count = [4412, 4862, 5806, 6536, 6980]  # 宠物猫数量（万只）
dog_count = [5503, 5222, 5429, 5119, 5175]  # 宠物狗数量（万只）
GDP_per_capita = [10143, 10408, 12617, 12662, 12614]  # 人均GDP（美元）
urbanization_rate = [44.38, 45.4, 46.7, 47.7, 48.3]  # 城镇化率（%）
total_population = [1409.67, 1411.75, 1412.60, 1412.12, 1410.08]  # 总人口（百万人）
cat_market_size = [798.77, 884, 1060, 1231, 1305]  # 猫的市场规模（亿元）
dog_market_size = [1210.66, 1181, 1430, 1475, 1488]  # 狗的市场规模（亿元）
cat_annual_consumption = [1810, 1818, 1826, 1883, 1870]  # 猫的单只年均消费金额（元）
dog_annual_consumption = [2200, 2262, 2634, 2882, 2875]  # 狗的单只年均消费金额（元）
 
# 初始值设置
E_0 = 100  # 初始出口量
D_0 = 500  # 初始全球需求
T_0 = 0.1  # 初始关税水平
P_0 = 50  # 初始出口价格
 
# 存储结果
E = np.zeros(years)
D = np.zeros(years)
T = np.zeros(years)
P = np.zeros(years)
 
# 初始值
E[0] = E_0
D[0] = D_0
T[0] = T_0
P[0] = P_0
 
# 随机过程生成
np.random.seed(42)
 
# 模拟过程
for t in range(1, years):
    # 关税随机波动
    epsilon_T = norm.rvs(loc=0, scale=sigma_T)
    T[t] = T[t - 1] + epsilon_T
 
    # 全球需求随机波动
    epsilon_D = norm.rvs(loc=0, scale=sigma_D)
    D[t] = alpha_1 * D[t - 1] + beta_1 * np.random.uniform(0, 0.05) + 0.1 * urbanization_rate[t] + 0.2 * GDP_per_capita[
        t] + 0.1 * (cat_count[t] + dog_count[t]) + epsilon_D
 
    # 关税对出口量的影响
    E[t] = E[t - 1] * (1 - gamma * T[t]) ** alpha_2 * (1 + lambda_ * T[t]) ** beta_2 * (1 + mu * D[t]) * (
                1 + 0.05 * (cat_count[t] + dog_count[t]))
 
    # 价格调整
    P[t] = rho * (1 + lambda_ * T[t]) + phi * (1 + delta * D[t]) + tau * np.sum(P[:t] - P[t]) + 0.05 * \
           cat_annual_consumption[t] + 0.1 * dog_annual_consumption[t]
 
# 可视化结果
plt.figure(figsize=(12, 6))
 
# 出口量
plt.subplot(2, 2, 1)
plt.plot(E, label='Export Volume')
plt.title('Export Volume Over Time')
plt.xlabel('Years')
plt.ylabel('Export Volume (in million USD)')
plt.grid(True)
 
# 全球需求
plt.subplot(2, 2, 2)
plt.plot(D, label='Global Demand', color='orange')
plt.title('Global Demand Over Time')
plt.xlabel('Years')
plt.ylabel('Global Demand (in million USD)')
plt.grid(True)
 
# 关税
plt.subplot(2, 2, 3)
plt.plot(T, label='Tariff Rate', color='red')
plt.title('Tariff Rate Over Time')
plt.xlabel('Years')
plt.ylabel('Tariff Rate')
plt.grid(True)
 
# 出口价格
plt.subplot(2, 2, 4)
plt.plot(P, label='Export Price', color='green')
plt.title('Export Price Over Time')
plt.xlabel('Years')
plt.ylabel('Export Price (in USD per ton)')
plt.grid(True)
 
plt.tight_layout()
plt.show()