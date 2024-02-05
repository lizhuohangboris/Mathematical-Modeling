import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# 提取特征
X = data[features]

# 将时间序列变量elapsed_time转化为秒
X["elapsed_time"] = pd.to_timedelta(data["elapsed_time"]).dt.total_seconds()

# 选择一个样本进行绘制
sample = X.iloc[0]

# 设置雷达图的标签
labels = features

# 角度
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()

# 闭合图形
values = sample.tolist()
values = values[:-1]  # 修改这里
angles += angles[:1]

# 绘制雷达图
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values, 'o-', linewidth=2, label="Player 1")
ax.fill(angles, values, alpha=0.25)

# 添加标签
ax.set_thetagrids(np.degrees(angles), labels)
ax.legend(loc="upper right")

plt.title("Radar Chart for Player 1")
plt.show()
