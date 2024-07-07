import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# 确认文件路径正确无误
train_file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\第二问.csv'
test_file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\test.csv'
output_file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024 亚太中文\洪水概率预测.csv'

# 加载训练数据，尝试使用逗号作为分隔符
train_data = pd.read_csv(train_file_path, delimiter=',', encoding='gbk')

# 加载预测数据
test_data = pd.read_csv(test_file_path, delimiter=',', encoding='gbk')

# 查看列名，去除可能存在的空格
train_data.columns = train_data.columns.str.strip()
test_data.columns = test_data.columns.str.strip()

# 分离特征和目标变量
X_train = train_data.drop('洪水概率', axis=1)
y_train = train_data['洪水概率']

# 确保训练数据和测试数据使用相同的特征列
common_features = [col for col in X_train.columns if col in test_data.columns]
X_train = X_train[common_features]
X_test = test_data[common_features]

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化并训练LightGBM回归模型
model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 进行预测
y_pred = model.predict(X_test_scaled)

# 将预测结果存储到新的DataFrame中
output_df = test_data.copy()
output_df['洪水概率'] = y_pred

# 输出预测结果到新的CSV文件
output_df.to_csv(output_file_path, index=False, encoding='gbk')

print(f'预测结果已保存到 {output_file_path}')
