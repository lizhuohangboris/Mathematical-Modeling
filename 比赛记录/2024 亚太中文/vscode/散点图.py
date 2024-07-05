import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 读取CSV文件，指定编码为'gbk'
def read_data(file_path, encoding='gbk'):
    try:
        data = pd.read_csv(file_path, encoding=encoding)
        return data
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在，请检查路径。")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return None

# 归一化处理
def normalize_data(data):
    data = data.drop(columns=['id'], errors='ignore')
    numeric_data = data.select_dtypes(include=[np.number])
    data_normalized = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())
    return data_normalized

# 可视化归一化后的数据
def plot_normalized_scatter(data_normalized, sample_size=100):
    plt.figure(figsize=(12, 8))
    sample_data = data_normalized.sample(n=sample_size, random_state=42)
    for column in sample_data.columns:
        plt.scatter(sample_data.index, sample_data[column], label=column, alpha=0.7, s=30)
    plt.title(f'归一化数据散点图 (样本数: {sample_size})', fontproperties=font)
    plt.xlabel('样本', fontproperties=font)
    plt.ylabel('归一化数值', fontproperties=font)
    plt.legend(prop=font)
    plt.show()

# 设置中文字体
font_path = 'C:\Windows\Fonts\simhei.ttf'  # 替换为你的中文字体文件路径
font = FontProperties(fname=font_path, size=14)

# 示例使用
if __name__ == "__main__":
    file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\train（箱形图法进行数据预处理）.csv'  # 替换为你的文件路径
    
    data = read_data(file_path)
    
    if data is not None:
        data_normalized = normalize_data(data)
        plot_normalized_scatter(data_normalized, sample_size=500)  # 选择500个样本进行展示
