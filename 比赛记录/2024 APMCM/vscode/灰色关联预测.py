import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei']  # 使用微软雅黑字体
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 文件路径
file_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 APMCM\数据\灰色关联度2.xlsx"

# 读取Excel数据
data = pd.read_excel(file_path)

# 设置因变量
dependent_variables = ["Cat（万只）", "Dog（万只）", "宠物行业市场规模", "宠物食物市场规模"]

# 定义 GM(1,1) 灰色预测模型
def gm11_model(series, forecast_years=3):
    n = len(series)
    
    if n < 2:
        raise ValueError("时间序列数据太短，无法进行灰色预测，请提供至少两个数据点")
    
    x1 = series.cumsum()  # 累加生成序列
    z1 = (x1[:-1] + x1[1:]) / 2  # 紧邻均值序列
    B = np.column_stack((-z1, np.ones(n - 1)))  # 构造 B 矩阵
    Y = series[1:]  # 构造 Y 向量
    
    # 检查 B 和 Y 的形状
    print(f"B 矩阵形状: {B.shape}, Y 向量形状: {Y.shape}")
    
    # 最小二乘法求解 a 和 b
    try:
        coefficients = np.linalg.inv(B.T @ B) @ B.T @ Y  # 求解参数
    except np.linalg.LinAlgError as e:
        raise ValueError(f"矩阵不可逆，无法进行灰色预测，可能是输入数据有问题：{e}")
    
    # 解包系数
    a, b = coefficients
    
    # 定义预测公式
    def predict(t):
        return (series[0] - b / a) * np.exp(-a * t) + b / a

    # 生成预测序列
    forecast = [predict(t) for t in range(n + forecast_years)]  # 预测未来序列
    forecast = np.diff(forecast, prepend=0)  # 还原为原序列
    return forecast

# 提取因变量的时间序列数据
years_column_name = "Years"  # 替换为实际年份列名
if years_column_name not in data.columns:
    raise KeyError(f"未找到年份列 '{years_column_name}'，请检查数据！")

years = data[years_column_name]
dependent_data = data[dependent_variables]

# 存储预测结果
predictions = pd.DataFrame({"Years": list(years) + [years.iloc[-1] + i for i in range(1, 4)]})

# 进行预测
for col in dependent_variables:
    series = dependent_data[col].values  # 当前因变量数据
    forecast = gm11_model(series)  # 预测未来三年
    predictions[col] = forecast

# 打印预测结果
print("未来三年的因变量预测值：")
print(predictions)

# 可视化预测结果
plt.figure(figsize=(12, 8))

for col in dependent_variables:
    plt.plot(predictions["Years"], predictions[col], marker='o', label=col)
    
    # 添加数据标点
    for x, y in zip(predictions["Years"], predictions[col]):
        plt.text(x, y, f"{y:.2f}", fontsize=9, ha='center', va='bottom')

plt.xlabel("年份", fontsize=14)
plt.ylabel("数值", fontsize=14)
plt.title("未来三年因变量预测趋势", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# 保存预测结果
output_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 APMCM\数据\Gray_Relation_Prediction_Result.xlsx"
predictions.to_excel(output_path, index=False)
print(f"预测结果已保存至 {output_path}")
