import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 加载数据
file_path = r'D:\Normal_tools\Github_desktop\Clone_shop\Mathematical-Modeling\比赛记录\2024亚太中文\train.csv'
data = pd.read_csv(file_path, encoding='gbk')

# 选择特征和目标变量
features = ['季风强度', '地形排水', '河流管理', '森林砍伐', '城市化', '气候变化', 
            '大坝质量', '淤积', '农业实践', '侵蚀', '无效防灾', '排水系统', '海岸脆弱性', 
            '滑坡', '流域', '基础设施恶化', '人口得分', '湿地损失', '规划不足', '政策因素']
target = '洪水概率'

X = data[features]
y = data[target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练决策树模型
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# 预测并评估模型
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# 提取特征重要性
feature_importance = model.feature_importances_
features_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
features_df = features_df.sort_values(by='Importance', ascending=False)

# 绘制特征重要性柱状图
plt.figure(figsize=(12, 8))
plt.bar(features_df['Feature'], features_df['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Decision Tree Model')
plt.show()
