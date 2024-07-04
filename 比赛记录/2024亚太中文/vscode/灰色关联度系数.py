import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    numeric_data = data.select_dtypes(include=[np.number])
    data_normalized = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())
    return data_normalized

# 计算灰色关联度系数
def grey_relational_coefficient(reference_series, comparison_series):
    diff_series = np.abs(reference_series - comparison_series)
    min_diff = diff_series.min().min()
    max_diff = diff_series.max().max()
    rho = 0.5
    return (min_diff + rho * max_diff) / (diff_series + rho * max_diff)

# 计算灰色关联度
def grey_relational_degree(data_normalized, reference_column):
    reference_series = data_normalized[reference_column]
    grey_rel_matrix = data_normalized.apply(lambda x: grey_relational_coefficient(reference_series, x))
    return grey_rel_matrix.mean()

# 主程序
def main():
    file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024亚太中文\train.csv'
    data = read_data(file_path)
    
    if data is not None:
        data_normalized = normalize_data(data)
        
        # 计算与洪水概率的灰色关联度
        grey_rel_degree = grey_relational_degree(data_normalized, '洪水概率')
        
        # 构造相关性矩阵
        correlation_matrix = pd.DataFrame(grey_rel_degree, columns=['灰色关联度'])
        
        # 打印每个变量的灰色关联度系数
        print(correlation_matrix)
        
        # 设置字体属性
        font_properties = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=8)  # 使用适当的中文字体路径
        
        # 设置绘图区域大小
        plt.figure(figsize=(16, 12))
        
        # 绘制热力图
        sns.heatmap(correlation_matrix.T, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5,
                    annot_kws={"size": 8, "fontproperties": font_properties})
        
        # 设置热力图标题
        plt.title('灰色关联度矩阵热力图', fontproperties=font_properties, size=16)
        
        # 调整刻度标签的字体属性
        plt.xticks(fontproperties=font_properties)
        plt.yticks(fontproperties=font_properties)
        
        # 显示热力图
        plt.show()

if __name__ == "__main__":
    main()
