import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'sans-serif']

# 读取数据集
file_path = r'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/数据(2).xlsx'
data = pd.read_excel(file_path)

# 检查数据是否平稳
def adfuller_test(series, signif=0.05):
    result = adfuller(series)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label + " : " + str(value))
    if result[1] <= signif:
        print("强证据拒绝原假设，数据平稳")
        return True
    else:
        print("弱证据接受原假设，数据非平稳")
        return False

# 检查每个列的平稳性
for col in data.columns[1:]:
    print("列名：" + col)
    adfuller_test(data[col])
    print("\n")

# 创建VAR模型并拟合数据
model = VAR(data)
model_fit = model.fit()

# 检查模型阶数
model_order = model_fit.k_ar
print('VAR模型阶数:', model_order)

# 执行预测
forecast = model_fit.forecast(model_fit.endog, steps=38)
forecast_df = pd.DataFrame(forecast, index=range(data.shape[0], data.shape[0]+38), columns=data.columns)

# 输出预测结果
print(forecast_df)

# 保存预测结果为Excel文件
output_path = 'forecast_results.xlsx'
forecast_df.to_excel(output_path, index=False)
print(f'预测结果已保存为 {output_path}')


# # 绘制可视化结果
# plt.figure(figsize=(10, 6))
# for col in data.columns[1:]:
#     plt.plot(data.index, data[col], label=col + ' (实际)', linestyle='-')
#     plt.plot(forecast_df.index, forecast_df[col], label=col + ' (预测)', linestyle='--')

# plt.title('VAR 模型预测结果')
# plt.xlabel('时间')
# plt.ylabel('值')
# plt.legend()
# plt.grid(True)
# plt.show()
# 绘制可视化结果



# plt.figure(figsize=(10, 6))
# plt.plot(data.index, data['二氧化碳排放量（百万吨）'], label='二氧化碳排放量 (实际)', linestyle='-')
# plt.plot(forecast_df.index, forecast_df['二氧化碳排放量（百万吨）'], label='二氧化碳排放量 (预测)', linestyle='--')

# plt.title('二氧化碳排放量随时间的预测')
# plt.xlabel('时间')
# plt.ylabel('二氧化碳排放量 (百万吨)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # 计算评估指标
# nobs = data.shape[0]
# df_modelwc = model_fit.df_modelwc
# print("RMSE:", rmse(data.iloc[-8:, 1:], forecast_df.iloc[:, 1:]))
# print("AIC:", aic(model_fit, nobs=nobs, df_modelwc=df_modelwc))

# 绘制可视化结果
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['二氧化碳排放量（百万吨）'], label='二氧化碳排放量 (实际)', linestyle='-')
plt.plot(forecast_df.index, forecast_df['二氧化碳排放量（百万吨）'], label='二氧化碳排放量 (预测)', linestyle='--')

# 在需要标注的点上绘制散点图并标注
plt.scatter(data.index[-1], data['二氧化碳排放量（百万吨）'].iloc[-1], color='red', label='最后一个观察点')
plt.annotate('最后一个观察点', (data.index[-1], data['二氧化碳排放量（百万吨）'].iloc[-1]), textcoords="offset points", xytext=(-10,-15), ha='center')

# 对真实值进行散点图标记和标注
for i, value in enumerate(data['二氧化碳排放量（百万吨）']):
    plt.scatter(data.index[i], value, color='blue', marker='o', label='实际值' if i == 0 else None)
    # plt.annotate(f'真实值: {value:.2f}', (data.index[i], value), textcoords="offset points", xytext=(10,5), ha='center', fontsize=8)

plt.title('二氧化碳排放量随时间的预测')
plt.xlabel('时间')
plt.ylabel('二氧化碳排放量 (百万吨)')
plt.legend()
plt.grid(True)
plt.show()

