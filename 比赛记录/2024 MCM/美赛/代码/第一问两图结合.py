import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load data from '数据处理.csv' with specific columns
file_path = "c:/Users/92579/Documents/GitHub/Mathematical-Modeling/比赛记录/2024 MCM/美赛/2024_MCM-ICM_Problems/2024_MCM-ICM_Problems/数据处理.csv"
columns_to_read = ['match_id', 'player1', 'player2', 'elapsed_time', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games',
                   'score_lead', 'Tie_breakers', 'server', 'serve_no', 'point_victor', 'game_victor', 'set_victor',
                   'p1_ace', 'p2_ace', 'p1_winner', 'p2_winner', 'p1_double_fault', 'p2_double_fault', 'p1_unf_err',
                   'p2_unf_err', 'p1_net_pt', 'p2_net_pt', 'p1_net_pt_won', 'p2_net_pt_won', 'p1_break_pt', 'p2_break_pt',
                   'p1_break_pt_won', 'p2_break_pt_won', 'p1_break_pt_missed', 'p2_break_pt_missed', 'p1_distance_run',
                   'p2_distance_run', 'rally_count']
df = pd.read_csv(file_path, usecols=columns_to_read)

# Choose features and target variable
features = ['elapsed_time', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games']
target = 'point_victor'

# Convert 'elapsed_time' to string and then to timedelta
df['elapsed_time'] = pd.to_timedelta(df['elapsed_time'].astype(str)).dt.total_seconds()

# Process other non-numeric features (dummy encoding)
df = pd.get_dummies(df, columns=['server', 'serve_no'])

# Encode 'point_victor' column
le = LabelEncoder()
df[target] = le.fit_transform(df[target])

# Split data into training and testing sets
train_size = 0.7
train, test = train_test_split(df, test_size=1 - train_size, random_state=42)

# Train Random Forest model on the first 70% of data
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(train[features], train[target])

# Use Random Forest model for predictions on the entire dataset
df['rf_predictions'] = rf_model.predict_proba(df[features])[:, 0]

# Inverse transform 'point_victor' for interpretation
df['point_victor'] = le.inverse_transform(df['point_victor'])

# Sort DataFrame by 'elapsed_time' for time series plot
df.sort_values('elapsed_time', inplace=True)

# Train LightGBM model on the first 70% of data
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(train[features], train[target])

# Use LightGBM model for predictions on the entire dataset
df['lgb_predictions'] = rf_model.predict_proba(df[features])[:, 1]

# Inverse transform 'point_victor' for interpretation
df['point_victor'] = le.inverse_transform(df['point_victor'])

# Sort DataFrame by 'elapsed_time' for time series plot
df.sort_values('elapsed_time', inplace=True)

# Plot the time series of predicted probabilities for Random Forest and LightGBM
plt.figure(figsize=(10, 6))
plt.plot(df['elapsed_time'], df['rf_predictions'], label='RF Predicted Probability', linestyle='dashed', color='skyblue')
plt.plot(df['elapsed_time'], df['lgb_predictions'], label='LightGBM Predicted Probability', linestyle='dashed', color='salmon')
# Scatter plot of 'elapsed_time' with predicted probabilities
plt.scatter(df['elapsed_time'], df['rf_predictions'], marker='o', s=3, color='blue', label='RF Scatter Plot of Elapsed Time')
plt.scatter(df['elapsed_time'], df['lgb_predictions'], marker='o', s=3, color='red', label='LightGBM Scatter Plot of Elapsed Time')
# Scatter plot of original 'elapsed_time' with 'point_victor' (0 and 1)
plt.scatter(df['elapsed_time'], df['point_victor'], marker='o', s=5, color='black', label='Original Data Points')

plt.xlabel('Elapsed Time')
plt.ylabel('Predicted Probability / Point Victor (0 or 1)')
plt.title('Visualization of the Momentum in Final Match (RF & LightGBM)')
plt.legend(bbox_to_anchor=(0.84

, 0.8), fontsize='small')

plt.show()
