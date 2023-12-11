import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# 读取Excel文件
data = pd.read_excel('C:/Users/92579/Desktop/6月.xlsx')

# 提取所需的数据列，例如：日期、销售量、售价、进货价格等
dates = data['销售日期']
sales_quantity = data[['品类1销售量', '品类2销售量', '品类3销售量', '品类4销售量', '品类5销售量', '品类6销售量']]
sales_price = data[['品类1售价', '品类2售价', '品类3售价', '品类4售价', '品类5售价', '品类6售价']]
purchase_price = data[['品类1进货价格', '品类2进货价格', '品类3进货价格', '品类4进货价格', '品类5进货价格', '品类6进货价格']]

# 计算每日销售额
daily_sales_revenue = sales_quantity * sales_price

# 每日总销售额
daily_total_sales = daily_sales_revenue.sum(axis=1)

# 定义目标函数：计算总利润
def calculate_total_profit(sales_price_func, purchase_price_func):
    daily_sales_revenue = sales_price_func(daily_sales_revenue)
    total_profit = daily_sales_revenue * (sales_price_func(daily_sales_revenue) - purchase_price_func(dates))
    return -total_profit.sum()

# 用线性规划拟合每日销售量和每日售价的函数
sales_price_functions = []
for col in sales_price.columns:
    def sales_price_optimizer(x):
        return calculate_total_profit(lambda x: x, interp1d(sales_quantity[col], sales_price[col], fill_value="extrapolate"))
    
    result = minimize(sales_price_optimizer, 1.0)  # Initial guess for the sales price function
    sales_price_func = interp1d(sales_quantity[col], sales_price[col], fill_value="extrapolate")
    sales_price_functions.append(sales_price_func)

# 用进货价格数据和日期拟合进货价格和日期的函数
purchase_price_functions = []
for col in purchase_price.columns:
    purchase_price_func = interp1d(dates, purchase_price[col], fill_value="extrapolate")
    purchase_price_functions.append(purchase_price_func)

# 预测2023年7月1日到7月7日的进货价格
forecast_dates = pd.date_range(start='2023-07-01', end='2023-07-07')
forecast_purchase_prices = []

for func in purchase_price_functions:
    forecast_purchase_prices.append(func(forecast_dates))

# 计算未来7月1日~7月7日6个品类蔬菜分别总利润最大时售价与时间的折线图
profit_max_sales_prices = []

for i in range(6):
    def profit_optimizer(x):
        return calculate_total_profit(sales_price_functions[i], interp1d(dates, x, fill_value="extrapolate"))
    
    result = minimize(profit_optimizer, purchase_price.iloc[0, i])  # Initial guess for purchase price
    profit_max_sales_prices.append(result.x)

# 绘制折线图
for i in range(6):
    plt.plot(forecast_dates, profit_max_sales_prices[i], label=f'品类{i+1}')

plt.xlabel('日期')
plt.ylabel('售价')
plt.legend()
plt.show()