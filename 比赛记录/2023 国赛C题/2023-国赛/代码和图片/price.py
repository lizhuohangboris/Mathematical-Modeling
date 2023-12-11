import pandas as pd
from scipy.optimize import linprog

# 从Excel读取数据
excel_file_path = 'C:/Users/92579/Desktop/6月.xlsx'
data = pd.read_excel(excel_file_path)

# 计算成本
data['成本'] = data['批发价格'] * (1 + data['单品损耗率'])

# 按大类分组
grouped = data.groupby('大类')

# 存储最优进货价格的列表
optimal_wholesale_prices = []

# 循环处理每个大类
for group_name, group_data in grouped:
    profits = (group_data['销售单价(元/千克)'] - group_data['成本']) * group_data['销量(千克)']
    wholesale_prices = group_data['批发价格']
    c = -profits
    costs = wholesale_prices * (1 + group_data['单品损耗率'])
    A_eq = [[1] * len(profits)]
    b_eq = [costs.sum()]
    bounds = [(0, None)] * len(profits)
    
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    optimal_wholesale_prices.append(result.x)

# 输出每个大类的最优进货价格
for group_name, prices in zip(grouped.groups.keys(), optimal_wholesale_prices):
    print(f"Optimal Wholesale Prices for {group_name}:")
    for i, price in enumerate(prices):
        print(f"Day {i + 1}: {price}")

# 计算销售日期为2023-6的总利润
total_profit_june_2023 = data[data['销售日期'].dt.month == 6]['总利润'].sum()
print(f"Total Profit in June 2023: {total_profit_june_2023}")

# 预测7月1日到7月7日的进货价格
predicted_wholesale_prices = []
for i in range(7):
    predicted_wholesale_prices.append(optimal_wholesale_prices[0][i])  # 假设这里选择第一个大类的最优价格

# 输出预测的进货价格
print("Predicted Wholesale Prices for July 1-7, 2023:")
for i, price in enumerate(predicted_wholesale_prices):
    print(f"July {i + 1}, 2023: {price}")
