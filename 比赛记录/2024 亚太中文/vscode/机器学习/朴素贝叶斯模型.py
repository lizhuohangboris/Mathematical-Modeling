import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, log_loss

# 确认文件路径正确无误
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\第二问.csv'

# 加载数据，尝试使用逗号作为分隔符
data = pd.read_csv(file_path, delimiter=',', encoding='gbk')

# 查看列名，去除可能存在的空格
data.columns = data.columns.str.strip()

# 分离特征和目标变量
X = data.drop('洪水概率', axis=1)
y = data['洪水概率']

# 将目标变量分成多个类别 (例如，将概率划分为10个区间)
num_bins = 10
y_binned = pd.cut(y, bins=num_bins, labels=False)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)

# 初始化并训练朴素贝叶斯模型
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# 预测概率
y_prob_nb = nb_model.predict_proba(X_test)

# 预测标签
y_pred_nb = nb_model.predict(X_test)

# 将预测的类别转换回概率值 (即，将区间标签转换回原始范围)
y_pred_prob = (y_pred_nb + 0.5) / num_bins

# 评估模型
mse_nb = mean_squared_error(y_test, y_pred_nb)
r2_nb = r2_score(y_test, y_pred_nb)
log_loss_nb = log_loss(y_test, y_prob_nb)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

print(f'Mean Squared Error (Naive Bayes): {mse_nb}')
print(f'R2 Score (Naive Bayes): {r2_nb}')
print(f'Log Loss (Naive Bayes): {log_loss_nb}')
print(f'Accuracy (Naive Bayes): {accuracy_nb}')
