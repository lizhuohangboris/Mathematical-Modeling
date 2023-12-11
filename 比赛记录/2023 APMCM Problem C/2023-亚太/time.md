# 读取必要的库
library(readxl)
library(forecast)

# 读取 Excel 数据
data <- read_excel("C:/Users/92579/Desktop/MATH/第二题.xlsx")

# 将数据转换为时间序列
ts_data <- ts(data$Y, start = min(data$year), freq = 1)

# 建立 TSLM 模型
model <- tslm(ts_data ~ X1 + X2, data = data)

# 生成模型的摘要报告
summary(model)

# 预测未来值和置信区间
future_years <- seq(max(data$year) + 1, max(data$year) + 10)
future_data <- data.frame(X1 = mean(data$X1), X2 = mean(data$X2), year = future_years)
future_forecast <- forecast(model, newdata = future_data)

# 查看预测结果，包括点估计和置信区间
print(future_forecast)

# 绘制历史数据、拟合值、预测和未来预测
plot(ts_data, main = "Sales data and forecasting", xlab = "Year", ylab = "Sales")
lines(fitted(model), col = "blue")  # 拟合值
lines(future_forecast$mean, col = "red")  # 预测值

# 在图中添加图例
legend("topleft", legend = c("History data", "Fitted data", ""Predicted data),
       col = c("black", "blue", "red"), lty = 1)





library(readxl) library(forecast) # 读取Excel数据 data <- read_excel("C:/Users/ka'le/Desktop/第二题.xlsx") # 将数据转换为时间序列 ts_data <- ts(data$Y, start = min(data$year), freq = 1) # 建立TSLM模型 model <- tslm(ts_data ~ X1 + X2 data = data) # 生成报告 summary(model) # 预测模型 future_years <- seq(max(data$year) + 1, max(data$year) + 10) future_data <- data.frame(X1 = mean(data$X1), X2 = mean(data$X2), year = future_years) future_forecast <- forecast(model, newdata = future_data) # 绘制原始数据和预测结果的折线图 plot(ts_data, main = "Sales Data and Forecast", xlab = "Year", ylab = "Sales") lines(fitted(model), col = "blue") lines(future_forecast$mean, col = "red") # 预测十年后的数据并绘图 lines(future_years, future_forecast$mean, col = "green") legend("topleft", legend = c("Historical Data", "Fitted Values", "Forecast", "Future Prediction"), col = c("black", "blue", "red", "green"), lty = 1)



library(readxl)
library(forecast)

# 读取Excel数据
data <- read_excel("C:/Users/92579/Desktop/MATH/第二题.xlsx")

# 将数据转换为时间序列
ts_data <- ts(data$Y, start = min(data$year), freq = 1)

# 建立TSLM模型
model <- tslm(ts_data ~ X1 + X2 + year, data = data)

# 生成报告
summary(model)

# 预测模型
future_years <- seq(max(data$year) + 1, max(data$year) + 10)
future_data <- data.frame(X1 = rep(mean(data$X1), 10), X2 = rep(mean(data$X2), 10), year = future_years)
future_forecast <- forecast(model, newdata = future_data)

# 绘制原始数据和预测结果的折线图
plot(ts_data, main = "Sales Data and Forecast", xlab = "Year", ylab = "Sales")
lines(fitted(model), col = "blue")
lines(future_years, future_forecast$mean, col = "red")

# 创建完整的时间序列对象
complete_ts_data <- c(ts_data, future_forecast$mean)
complete_ts_years <- c(time(ts_data), future_years)
complete_ts <- ts(complete_ts_data, start = min(complete_ts_years), freq = 1)

# 绘制包括预测的完整数据折线图
plot(complete_ts, main = "Sales Data and Forecast", xlab = "Year", ylab = "Sales")
lines(ts_data, col = "black")
lines(future_years, future_forecast$mean, col = "green")
legend("topleft", legend = c("Historical Data", "Forecast", "Future Prediction"), col = c("black", "red", "green"), lty = 1)

![image-20231124222528394](C:\Users\92579\AppData\Roaming\Typora\typora-user-images\image-20231124222528394.png)





# 读取数据
data <- read.table(text = "-0.841749845 -0.683983205 -1.346830906 1
-0.838578417 -0.68068968 -1.243228528 2
-0.834614132 -0.67309441 -1.139626151 3
-0.789421283 -0.661936757 -0.828819019 4
-0.588035606 -0.649703667 -0.518011887 5
-0.450871345 -0.551435658 -0.207204755 6
-0.239971384 -0.38877589 0.103602377 7
0.139807117 -0.166294967 0.414409509 8
0.10729998 0.130794361 0.725216641 9
0.200064249 0.441326646 1.036023774 10
1.93166393 1.070456988 1.346830906 11
2.204406737 2.813336239 1.657638038 12", header = FALSE)

# 命名列
colnames(data) <- c("Y", "X1", "X2", "year")

# 划分数据集
train_data <- data[1:8, ]
test_data <- data[9:12, ]

# 创建线性回归模型
linear_model <- lm(Y ~ X1 + X2 + year, data = train_data)

# 模型预测
predictions <- predict(linear_model, newdata = test_data)

# 评估模型性能
rmse <- sqrt(mean((test_data$Y - predictions)^2))
mae <- mean(abs(test_data$Y - predictions))

# 打印评估指标
print(paste("RMSE:", rmse))
print(paste("MAE:", mae))





# 安装所需的包
install.packages("e1071")
install.packages("randomForest")

# 加载所需的包
library(e1071)
library(randomForest)

# 读取数据
data <- read.table(text = "-0.841749845 -0.683983205 -1.346830906 1
-0.838578417 -0.68068968 -1.243228528 2
-0.834614132 -0.67309441 -1.139626151 3
-0.789421283 -0.661936757 -0.828819019 4
-0.588035606 -0.649703667 -0.518011887 5
-0.450871345 -0.551435658 -0.207204755 6
-0.239971384 -0.38877589 0.103602377 7
0.139807117 -0.166294967 0.414409509 8
0.10729998 0.130794361 0.725216641 9
0.200064249 0.441326646 1.036023774 10
1.93166393 1.070456988 1.346830906 11
2.204406737 2.813336239 1.657638038 12", header = FALSE)

# 命名列
colnames(data) <- c("Y", "X1", "X2", "year")

# 划分数据集
train_data <- data[1:8, ]
test_data <- data[9:12, ]

# 创建支持向量机模型
svm_model <- svm(Y ~ X1 + X2 + year, data = train_data)

# 模型预测
svm_predictions <- predict(svm_model, newdata = test_data)

# 评估模型性能
svm_rmse <- sqrt(mean((test_data$Y - svm_predictions)^2))
svm_mae <- mean(abs(test_data$Y - svm_predictions))

# 打印支持向量机模型的评估指标
cat("SVM Model:\n")
cat("RMSE:", svm_rmse, "\n")
cat("MAE:", svm_mae, "\n\n")

# 创建随机森林模型
rf_model <- randomForest(Y ~ X1 + X2 + year, data = train_data)

# 模型预测
rf_predictions <- predict(rf_model, newdata = test_data)

# 评估模型性能
rf_rmse <- sqrt(mean((test_data$Y - rf_predictions)^2))
rf_mae <- mean(abs(test_data$Y - rf_predictions))

# 打印随机森林模型的评估指标
cat("Random Forest Model:\n")
cat("RMSE:", rf_rmse, "\n")
cat("MAE:", rf_mae, "\n")