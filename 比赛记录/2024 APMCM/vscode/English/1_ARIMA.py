import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# File path
file_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 APMCM\数据\灰色关联度2.xlsx"

# Read data
data = pd.read_excel(file_path)

# Define dependent variables
dependent_variables = ["Cat", "Dog", "Pet Industry Market Size"]

# Function to check stationarity
def check_stationarity(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    if result[1] > 0.05:
        print("The series is non-stationary. Differencing or transformation is recommended.")
    else:
        print("The series is stationary.")

# Check stationarity for all dependent variables
print("Stationarity Check Results:")
for col in dependent_variables:
    print(f"Checking stationarity for {col}:")
    check_stationarity(data[col])

# Define ARIMA forecast function
def arima_forecast(series, forecast_years=3):
    # Check and difference data if necessary
    if adfuller(series)[1] > 0.05:  # If non-stationary, perform first-order differencing
        series_diff = series.diff().dropna()
    else:
        series_diff = series

    # Fit ARIMA model
    model = ARIMA(series_diff, order=(1, 1, 0))  # Adjust (p, d, q) as needed
    fitted_model = model.fit()

    # Generate forecasts
    forecast_diff = fitted_model.forecast(steps=forecast_years)

    # Reverse differencing to restore original scale
    forecast_cumsum = forecast_diff.cumsum()
    forecast_final = forecast_cumsum + series.iloc[-1]

    # Restrict negative values
    forecast_final[forecast_final < 0] = 0
    return forecast_final

# Create a DataFrame for predictions
predictions = pd.DataFrame({"Years": list(data["Years"]) + [data["Years"].iloc[-1] + i for i in range(1, 4)]})

# Generate forecasts for each dependent variable
for col in dependent_variables:
    series = data[col]
    forecast = arima_forecast(series, forecast_years=3)
    predictions[col] = list(series) + list(forecast)

# Print forecast results
print("Forecast values for the next three years:")
print(predictions)

# Visualize forecast results
plt.figure(figsize=(12, 8))

# 绘制 Cat 和 Dog 的折线图
for col in dependent_variables:
    if col != "Pet Industry Market Size":  # 对 Cat 和 Dog 使用折线图
        plt.plot(predictions["Years"], predictions[col], marker='o', label=col)
        
        # Annotate data points
        for x, y in zip(predictions["Years"], predictions[col]):
            plt.text(x, y, f"{y:.2f}", fontsize=9, ha='center', va='bottom')

# 绘制 Pet Industry Market Size 的柱状图
plt.bar(predictions["Years"], predictions["Pet Industry Market Size"], color='skyblue', label="Pet Industry Market Size", alpha=0.7)

# Annotate柱状图上的数据点
for x, y in zip(predictions["Years"], predictions["Pet Industry Market Size"]):
    plt.text(x, y, f"{y:.2f}", fontsize=9, ha='center', va='bottom')

# 添加标题和标签
plt.xlabel("Year", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.title("Forecast Visualization with Mixed Chart Types", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

# 展示图形
plt.show()


# 输出每个模型的相关信息
print("\nModel Summary and Diagnostics:")
for col in dependent_variables:
    series = data[col]
    if adfuller(series)[1] > 0.05:  # 检查是否平稳，若非平稳则差分
        series_diff = series.diff().dropna()
    else:
        series_diff = series

    # 拟合 ARIMA 模型
    model = ARIMA(series_diff, order=(1, 1, 0))
    fitted_model = model.fit()

    # 打印模型信息
    print(f"\nVariable: {col}")
    print(fitted_model.summary())
    print("\nDiagnostics:")
    print(f"AIC: {fitted_model.aic}")
    print(f"BIC: {fitted_model.bic}")
    print(f"HQIC: {fitted_model.hqic}")  # Hannan-Quinn 信息准则
# 检查数据平稳性
print("Stationarity Check Results:")
for col in dependent_variables:
    print(f"Checking stationarity for {col}:")
    check_stationarity(data[col])



# Save forecast results
output_path = r"D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 APMCM\数据\ARIMA_Prediction_Result.xlsx"
predictions.to_excel(output_path, index=False)
print(f"Forecast results have been saved to {output_path}")
