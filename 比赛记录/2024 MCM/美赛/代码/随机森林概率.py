# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt

# # 加载数据
# file_path = "c:/Users/92579/Documents/GitHub/Mathematical-Modeling/比赛记录/2024 MCM/美赛/2024_MCM-ICM_Problems/2024_MCM-ICM_Problems/de.xlsx"
# df = pd.read_excel(file_path)

# # 选择特征和目标变量
# features = ['elapsed_time', 'set_no', 'game_no', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games']
# target = 'point_victor'

# # Convert 'elapsed_time' to string and then to timedelta
# df['elapsed_time'] = pd.to_timedelta(df['elapsed_time'].astype(str)).dt.total_seconds()

# # Continue with the rest of your code...


# # 处理其他非数值型特征，可以使用更复杂的方法进行处理
# df = pd.get_dummies(df, columns=['server', 'serve_no', 'winner_shot_type'])

# # 分割数据集为训练集和测试集
# train_size = 0.7
# train, test = train_test_split(df, test_size=1-train_size, random_state=42)

# # 建立和训练模型
# model = RandomForestClassifier(random_state=42)
# model.fit(train[features], train[target])

# # 在测试集上进行预测
# test['predicted_prob'] = model.predict_proba(test[features])[:, 1]

# # 绘制预测概率图
# plt.hist(test[test['point_victor'] == 1]['predicted_prob'], bins=50, alpha=0.5, label='Point Victor 1')
# plt.hist(test[test['point_victor'] == 2]['predicted_prob'], bins=50, alpha=0.5, label='Point Victor 2')
# plt.xlabel('Predicted Probability')
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# 读取数据
file_path = "c:/Users/92579/Documents/GitHub/Mathematical-Modeling/比赛记录/2024 MCM/美赛/2024_MCM-ICM_Problems/2024_MCM-ICM_Problems/数据处理.csv"
df = pd.read_excel(file_path, engine='openpyxl')

# 假设数据已经进行了一些处理，包括缺失值处理、分类变量转换等

# 选择特征列
features = ['elapsed_time', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'score_lead', 'Tie_breakers',
            'server', 'serve_no', 'p1_ace', 'p2_ace', 'p1_winner', 'p2_winner', 'p1_double_fault', 
            'p2_double_fault', 'p1_unf_err', 'p2_unf_err', 'p1_net_pt', 'p2_net_pt', 'p1_net_pt_won', 
            'p2_net_pt_won', 'p1_break_pt', 'p2_break_pt', 'p1_break_pt_won', 'p2_break_pt_won', 
            'p1_break_pt_missed', 'p2_break_pt_missed', 'p1_distance_run', 'p2_distance_run', 'rally_count']

# 选择因变量
target = 'point_victor'

# 划分训练集和测试集
train, test = train_test_split(df, test_size=0.3, random_state=42)

# 初始化决策树模型
model = DecisionTreeClassifier(random_state=42)

# 训练模型
model.fit(train[features], train[target])

# 预测概率
test['predicted_prob'] = model.predict_proba(test[features])[:, 1]

# 绘制预测概率图
plt.plot(test['elapsed_time'], test['predicted_prob'], label='Predicted Probability')
plt.xlabel('Elapsed Time')
plt.ylabel('Predicted Probability')
plt.title('Decision Tree Predicted Probability for Player A')
plt.legend()
plt.show()
