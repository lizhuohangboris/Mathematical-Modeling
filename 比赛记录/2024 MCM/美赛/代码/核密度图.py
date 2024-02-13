import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# 将时间序列变量elapsed_time转化为秒
X["elapsed_time"] = pd.to_timedelta(data["elapsed_time"]).dt.total_seconds()

# 绘制核密度估计图
plt.figure(figsize=(12, 8))
for feature in features:
    sns.kdeplot(X[feature], label=feature, fill=True)

plt.title("Kernel Density Estimation for Features")
plt.xlabel("Feature Values")
plt.ylabel("Density")
plt.legend()
plt.show()
