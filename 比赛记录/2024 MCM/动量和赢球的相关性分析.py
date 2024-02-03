import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import plot_tree

# Load data from '数据处理.csv' with specific columns
file_path = "c:/Users/92579/Documents/GitHub/Mathematical-Modeling/比赛记录/2024 MCM/美赛/2024_MCM-ICM_Problems/2024_MCM-ICM_Problems/数据处理.csv"
columns_to_read = ['match_id', 'player1', 'player2', 'elapsed_time', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games',
                   'score_lead', 'Tie_breakers', 'server', 'serve_no', 'point_victor', 'game_victor', 'set_victor',
                   'p1_ace', 'p2_ace', 'p1_winner', 'p2_winner', 'p1_double_fault', 'p2_double_fault', 'p1_unf_err',
                   'p2_unf_err', 'p1_net_pt', 'p2_net_pt', 'p1_net_pt_won', 'p2_net_pt_won', 'p1_break_pt', 'p2_break_pt',
                   'p1_break_pt_won', 'p2_break_pt_won', 'p1_break_pt_missed', 'p2_break_pt_missed', 'p1_distance_run',
                   'p2_distance_run', 'rally_count']
df = pd.read_csv(file_path, usecols=columns_to_read)

# Convert 'elapsed_time' to datetime
df['elapsed_time'] = pd.to_datetime(df['elapsed_time'])

# Choose features and target variable
features_rf = ['p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'score_lead']
features_xg = features_rf + ['rf_predictions']  # Include Random Forest predictions as a feature for XGBoost
target = 'point_victor'

# Encode 'point_victor' column
le = LabelEncoder()
df[target] = le.fit_transform(df[target])

# Split data into training and testing sets
train_size = 0.7
train, test = train_test_split(df, test_size=1 - train_size, random_state=42)

# Train Random Forest model on the first 70% of data
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(train[features_rf], train[target])

# Use Random Forest model for predictions on the entire dataset
df['rf_predictions'] = rf_model.predict(df[features_rf])

# Train XGBoost model on the entire dataset with 'rf_predictions' as a feature
xg_model = XGBClassifier(random_state=42)
xg_model.fit(df[features_xg], df[target])

# Use XGBoost model for predictions on the entire dataset
df['xg_predictions'] = xg_model.predict(df[features_xg])

# Calculate accuracy for XGBoost
xg_accuracy = accuracy_score(df[target], df['xg_predictions'])
print("XGBoost Accuracy:", xg_accuracy)

# Confusion matrix for XGBoost
xg_conf_matrix = confusion_matrix(df[target], df['xg_predictions'])
plt.figure(figsize=(8, 6))
sns.heatmap(xg_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('XGBoost Confusion Matrix')
plt.show()

# Use XGBoost model for predictions on the entire dataset
df['xg_probabilities'] = xg_model.predict_proba(df[features_xg])[:, 1]

# Inverse transform 'point_victor' for interpretation
df['point_victor'] = le.inverse_transform(df['point_victor'])

# Calculate accuracy for XGBoost
xg_accuracy = accuracy_score(df[target], df['xg_predictions'])
print("XGBoost Accuracy:", xg_accuracy)

# Plot the time series of predicted probabilities for XGBoost
plt.figure(figsize=(15, 8))

# Subplot 1: XGBoost Predicted Probability and Original Data Points
plt.plot(df['elapsed_time'], df['xg_probabilities'], label='XGBoost Predicted Probability (Player 1)', linestyle='dashed', color='salmon')
plt.scatter(df['elapsed_time'], df['point_victor'], marker='o', s=5, color='black', label='Original Data Points - XG')
plt.scatter(df['elapsed_time'], df['xg_probabilities'], marker='o', s=5, color='blue', label='Predicted Probabilities - XG')  # Blue points at predicted points
plt.xlabel('Elapsed Time')
plt.ylabel('Predicted Probability / Point Victor (0 or 1)')
plt.title('XGBoost - Time Series of Predicted Probability with Scatter Plot')
plt.legend()

# Extract relevant columns for correlation analysis
correlation_columns = ['rf_predictions', 'xg_probabilities']

# Calculate correlation matrix
correlation_matrix = df[correlation_columns].corr()
print("Pearson Correlation Coefficients:")
print(correlation_matrix)
# Visualize correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix: Random Forest Predictions vs. XGBoost Probabilities')
plt.show()

# 选择第一棵树
tree_index = 0

# 绘制生成数图
plt.figure(figsize=(20, 10))
plot_tree(xg_model, num_trees=tree_index, rankdir='LR')
plt.show()
