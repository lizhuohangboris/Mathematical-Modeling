import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def binomial_test(sample_size, defective_items, defective_rate, confidence_level):
    # """
    # 使用二项分布进行假设检验，判断是否接受这批零配件。
    # :param sample_size: 样本大小
    # :param defective_items: 样本中次品的数量
    # :param defective_rate: 标称次品率（如 10%）
    # :param confidence_level: 检验的置信水平
    # :return: 是否接收零配件
    # """
    # 计算显著性水平
    alpha = 1 - confidence_level
    
    # 使用新的 binomtest 进行检验，计算拒绝零假设的 p 值
    result = stats.binomtest(defective_items, sample_size, defective_rate, alternative='greater')

    # 判断是否拒绝零假设
    if result.pvalue < alpha:
        return f"拒收零配件 (p-value: {result.pvalue:.5f})"
    else:
        return f"接受零配件 (p-value: {result.pvalue:.5f})"

def calculate_min_sample_size(defective_rate, confidence_level, error_margin):
    # """
    # 计算最小样本量，满足置信水平和误差率要求
    # :param defective_rate: 标称次品率
    # :param confidence_level: 置信水平
    # :param error_margin: 允许的误差率
    # :return: 最小样本量
    # """
    # 根据置信水平计算 Z 分数
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    
    # 计算最小样本量的公式
    sample_size = (z_score**2 * defective_rate * (1 - defective_rate)) / (error_margin**2)
    
    # 返回向上取整后的样本量
    return int(np.ceil(sample_size))

# 设定标称次品率和置信水平
defective_rate = 0.10  # 标称次品率为 10%
confidence_level_95 = 0.95  # 95%的置信水平
confidence_level_90 = 0.90  # 90%的置信水平
error_margin = 0.05  # 允许的误差率为 5%

# 计算最小样本量
sample_size_95 = calculate_min_sample_size(defective_rate, confidence_level_95, error_margin)
sample_size_90 = calculate_min_sample_size(defective_rate, confidence_level_90, error_margin)

print(f"95% 置信水平下的最小样本量: {sample_size_95}")
print(f"90% 置信水平下的最小样本量: {sample_size_90}")

# 假设我们检测到样本中的次品数量
detected_defective_items = 12  # 假设样本中有 12 个次品

# 对两种置信水平进行二项分布假设检验
result_95 = binomial_test(sample_size_95, detected_defective_items, defective_rate, confidence_level_95)
result_90 = binomial_test(sample_size_90, detected_defective_items, defective_rate, confidence_level_90)

print(f"95% 置信水平结果: {result_95}")
print(f"90% 置信水平结果: {result_90}")
def plot_binomial_distribution(sample_size, defective_rate):
    """
    绘制二项分布图。
    :param sample_size: 样本大小
    :param defective_rate: 标称次品率
    """
    x = np.arange(0, sample_size + 1)
    probabilities = stats.binom.pmf(x, sample_size, defective_rate)
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, probabilities, color='skyblue', edgecolor='black')
    plt.xlabel('次品数量')
    plt.ylabel('概率')
    plt.title(f'二项分布概率图\n样本大小 = {sample_size}, 标称次品率 = {defective_rate}')
    plt.grid(axis='y')
    plt.show()

# 使用示例
plot_binomial_distribution(sample_size_95, defective_rate)
plot_binomial_distribution(sample_size_90, defective_rate)
