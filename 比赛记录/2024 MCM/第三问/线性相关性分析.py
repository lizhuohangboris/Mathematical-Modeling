import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 读取数据
data = pd.read_csv("c:/Users/92579/Documents/GitHub/Mathematical-Modeling/比赛记录/2024 MCM/美赛/2024_MCM-ICM_Problems/2024_MCM-ICM_Problems/数据处理.csv")

# 选择相关的自变量
features = ["p1_sets", "p2_sets", "p1_games", "p2_games", "score_lead", "Tie_breakers",
            "server", "serve_no", "game_victor", "set_victor", "p1_ace", "p2_ace", 
            "p1_winner", "p2_winner", "p1_double_fault", "p2_double_fault", 
            "p1_unf_err", "p2_unf_err", "p1_net_pt", "p2_net_pt", 
            "p1_net_pt_won", "p2_net_pt_won", "p1_break_pt", "p2_break_pt", 
            "p1_break_pt_won", "p2_break_pt_won", "p1_break_pt_missed", 
            "p2_break_pt_missed", "p1_distance_run", "p2_distance_run", "rally_count"]

# 提取特征和目标变量
X = data[features]
y = data["point_victor"]

# 将分类变量转化为数值型
X = pd.get_dummies(X)

# 将时间序列变量elapsed_time转化为秒
X["elapsed_time"] = pd.to_timedelta(data["elapsed_time"]).dt.total_seconds()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 查看模型的系数
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
print(coefficients)
