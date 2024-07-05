import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# 输出统计信息
def get_statistics(data_normalized):
    statistics = {
        'Mean': data_normalized.mean(),
        'Standard Deviation': data_normalized.std(),
        'Min': data_normalized.min(),
        'Max': data_normalized.max()
    }
    stats_df = pd.DataFrame(statistics)
    return stats_df

# 可视化统计信息
def plot_statistics(stats_df):
    plt.figure(figsize=(14, 8))
    markers = {'Mean': 'o', 'Standard Deviation': 's', 'Min': 'x', 'Max': 'd'}
    for stat in stats_df.columns:
        plt.scatter(stats_df.index, stats_df[stat], label=stat, s=50, marker=markers[stat])
    plt.title('Statistics of Normalized Data')
    plt.xlabel('Variables')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

# 示例使用
if __name__ == "__main__":
    file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\train（箱形图法进行数据预处理）.csv'  # 替换为你的文件路径
    
    data = read_data(file_path)
    
    if data is not None:
        data_normalized = normalize_data(data)
        stats_df = get_statistics(data_normalized)
        print(stats_df)
        plot_statistics(stats_df)
