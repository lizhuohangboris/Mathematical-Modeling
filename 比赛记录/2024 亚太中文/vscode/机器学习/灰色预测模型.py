import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error

# 确认文件路径正确无误
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\五个变量的数据.csv'

# 尝试使用不同的编码格式加载数据
encodings = [ 'gbk', 'utf-16']

for encoding in encodings:
    try:
        data = pd.read_csv(file_path, delimiter=',', encoding=encoding)
        print(f'Successfully loaded data using {encoding} encoding.')
        break
    except UnicodeDecodeError:
        print(f'Failed to load data using {encoding} encoding.')
        continue

# 查看列名，去除可能存在的空格
data.columns = data.columns.str.strip()

# 分离特征和目标变量
X = data.drop('洪水概率', axis=1)
y = data['洪水概率']

# 将目标变量转换为分类标签
y = y.astype('category').cat.codes

# 选取一个特征进行灰色预测建模（以第一个特征为例）
X_feature = X.iloc[:, 0].values

# 将数据分为训练集和测试集
train_size = int(len(X_feature) * 0.8)
X_train, X_test = X_feature[:train_size], X_feature[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 灰色预测模型函数
def GM11(x0):
    n = len(x0)
    x1 = np.cumsum(x0)
    B = np.zeros((n - 1, 2))
    Y = x0[1:].reshape(n - 1, 1)
    
    for i in range(n - 1):
        B[i][0] = -0.5 * (x1[i] + x1[i + 1])
        B[i][1] = 1.0
        
    B = np.mat(B)
    Y = np.mat(Y)
    
    # a, b = (BTB)^-1 BTY
    [[a], [b]] = (B.T * B).I * B.T * Y
    
    def f(k):
        return (x0[0] - b / a) * np.exp(-a * k) + b / a
    
    return [f(k) for k in range(n)], a, b

# 构建灰色预测模型并进行预测
predicted_train, a, b = GM11(X_train)

# 进行预测
predicted_test = [(X_train[0] - b / a) * np.exp(-a * (k + len(X_train))) + b / a for k in range(len(X_test))]

# 确保 y_test 和 predicted_test 是一维 NumPy 数组，并且形状一致
y_test = np.array(y_test).flatten()
predicted_test = np.array(predicted_test).flatten()

# 检查 y_test 和 predicted_test 的形状
print("Shape of y_test:", y_test.shape)
print("Shape of predicted_test:", predicted_test.shape)

# 评估模型
mse = mean_squared_error(y_test, predicted_test)
r2 = r2_score(y_test, predicted_test)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R² Score: {r2}')

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题

# 绘制预测结果对比图
plt.figure()
plt.plot(range(len(y_test)), y_test, 'b', label='实际值')
plt.plot(range(len(predicted_test)), predicted_test, 'r', label='预测值')
plt.xlabel('样本索引')
plt.ylabel('洪水概率')
plt.title('灰色模型预测结果对比')
plt.legend()
plt.show()
