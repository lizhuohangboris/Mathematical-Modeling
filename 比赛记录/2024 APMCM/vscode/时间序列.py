import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
 
# 示例：美国、法国、德国的宠物猫狗数量（单位：万只），数据来自题目中的附件2
# 数据按年份排列
data = {
    'Year': [2019, 2020, 2021, 2022, 2023],
    'USA_Cats': [9420, 6500, 9420, 7380, 7380],
    'USA_Dogs': [8970, 8500, 8970, 8970, 8010],
    'France_Cats': [1300, 1490, 1510, 1490, 1660],
    'France_Dogs': [740, 775, 750, 760, 990],
    'Germany_Cats': [1470, 1570, 1670, 1520, 1570],
    'Germany_Dogs': [1010, 1070, 1030, 1060, 1050]
}
 
# 将数据转换为 DataFrame
df = pd.DataFrame(data)
df.set_index('Year', inplace=True)
 
# 合并各国宠物数量为单一时间序列：假设关注全球猫和狗总数
total_cats = df['USA_Cats'] + df['France_Cats'] + df['Germany_Cats']
total_dogs = df['USA_Dogs'] + df['France_Dogs'] + df['Germany_Dogs']
 
# 合并成一个总的时间序列
total_pets = total_cats + total_dogs
 
# 进行ADF检验，检查时间序列是否平稳
def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("时间序列是平稳的")
    else:
        print("时间序列是非平稳的，需要差分")
 
# 检查总宠物数量是否平稳
check_stationarity(total_pets)
 
# 若非平稳，进行差分
total_pets_diff = total_pets.diff().dropna()
 
# 再次检查差分后的数据是否平稳
check_stationarity(total_pets_diff)
 
# 绘制ACF和PACF图以确定ARIMA模型参数
# 由于差分后的数据长度为4，lags的最大值应小于2
plot_acf(total_pets_diff, lags=2)  # 滞后期数改为2
plot_pacf(total_pets_diff, lags=2)  # 滞后期数改为2
plt.show()
 
 
# 根据ACF和PACF图的分析，假设选择p=1, d=1, q=1进行ARIMA建模
p, d, q = 1, 1, 1
 
# 构建并拟合ARIMA模型
model = ARIMA(total_pets, order=(p, d, q))
fitted_model = model.fit()
 
# 输出模型摘要
print(fitted_model.summary())
 
# 使用拟合模型进行未来3年的预测
forecast_steps = 3
forecast = fitted_model.forecast(steps=forecast_steps)
 
# 输出预测结果
forecast_years = [2024, 2025, 2026]
forecast_df = pd.DataFrame({
    'Year': forecast_years,
    'Predicted_Pets': forecast
})
 
# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(df.index, total_pets, label='Historical Pet Numbers')
plt.plot(forecast_df['Year'], forecast_df['Predicted_Pets'], label='Forecasted Pet Numbers', marker='o', linestyle='--', color='red')
plt.xlabel('Year')
plt.ylabel('Total Number of Pets (10,000s)')
plt.title('Total Pets (Cats + Dogs) in USA, France, Germany: Historical and Forecasted')
plt.legend()
plt.grid(True)
plt.show()
 
# 假设每只宠物每年需要k单位的宠物食品（例如，k = 1个单位食品需求）
k = 1
forecast_df['Predicted_Food_Consumption'] = forecast_df['Predicted_Pets'] * k
 
# 输出预测的宠物食品需求
print(forecast_df[['Year', 'Predicted_Food_Consumption']])