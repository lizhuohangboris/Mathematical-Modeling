import inspect
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR

# 读取数据
file_path = r'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/数据(2).xlsx'
data = pd.read_excel(file_path)
# 选择包含所有因素的列
factors = ['GDP（十亿）', '人口（百万人）', '二氧化碳排放量（百万吨）', '能源消耗', '钢铁产量（千吨）', 
           '水泥（百万吨）', '民用汽车数量（千辆）', '煤炭消耗量（百万吨）', '原油消耗量', 
           '天然气消耗量', '新能源消耗量']

# 从数据中提取包含所有因素的列
data = data[factors]

# 模型拟合
model = VAR(data)
model_fit = model.fit()

# 准备预测用的起始点
start_point = data.values[-1]  # 使用数据集中的最后一行作为起始点

# 打印一些中间变量
print("start_point shape:", start_point.shape)
print("model_fit.exog_names:", model_fit.exog_names)
# 预测未来的数据
forecast_2030 = model_fit.forecast(steps=2030 - 2022, exog_future=None)




print(forecast_2030)

