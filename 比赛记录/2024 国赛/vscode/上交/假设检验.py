import numpy as np
import scipy.stats as stats

def binomial_test(sample_size, defective_items, defective_rate, confidence_level):
    alpha = 1 - confidence_level
    result = stats.binomtest(defective_items, sample_size, defective_rate, alternative='greater')
    if result.pvalue < alpha:
        return f"拒绝 (p-value: {result.pvalue:.5f})"
    else:
        return f"接受 (p-value: {result.pvalue:.5f})"
def calculate_min_sample_size(defective_rate, confidence_level, error_margin):
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    sample_size = (z_score**2 * defective_rate * (1 - defective_rate)) / (error_margin**2)
    return int(np.ceil(sample_size))
defective_rate = 0.10  
confidence_level_95 = 0.95  
confidence_level_90 = 0.90 
error_margin = 0.05  # 假设误差率： 5%
sample_size_95 = calculate_min_sample_size(defective_rate, confidence_level_95, error_margin)
sample_size_90 = calculate_min_sample_size(defective_rate, confidence_level_90, error_margin)
print(f"95% 置信水平下的最小样本量: {sample_size_95}")
print(f"90% 置信水平下的最小样本量: {sample_size_90}")