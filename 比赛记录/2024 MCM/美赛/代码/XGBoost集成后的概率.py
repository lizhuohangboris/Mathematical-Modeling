import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
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
features_rf = ['elapsed_time', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'score_lead']
features_xg = ['elapsed_time', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'score_lead', 'rf_predictions']
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
rf_model.fit(train[features_rf], train[target])

# Use Random Forest model for predictions on the test dataset
test['rf_predictions'] = rf_model.predict_proba(test[features_rf])[:, 1]

# Add 'rf_predictions' column to the train dataset
train['rf_predictions'] = rf_model.predict_proba(train[features_rf])[:, 1]

# Train XGBoost model on the first 70% of data with 'score_lead' and 'rf_predictions'
xg_model = XGBClassifier(random_state=42)
xg_model.fit(train[features_xg], train[target])

# Use XGBoost model for predictions on the test dataset
test['xg_predictions'] = xg_model.predict_proba(test[features_xg])[:, 1]

# Calculate accuracy for Random Forest
rf_accuracy = accuracy_score(test[target], (test['rf_predictions'] > 0.5).astype(int))
print("Random Forest Accuracy:", rf_accuracy)

# Calculate accuracy for XGBoost
xg_accuracy = accuracy_score(test[target], (test['xg_predictions'] > 0.5).astype(int))
print("XGBoost Accuracy:", xg_accuracy)

# Confusion matrix for Random Forest
rf_conf_matrix = confusion_matrix(test[target], (test['rf_predictions'] > 0.5).astype(int))
plt.figure(figsize=(8, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Confusion matrix for XGBoost
xg_conf_matrix = confusion_matrix(test[target], (test['xg_predictions'] > 0.5).astype(int))
plt.figure(figsize=(8, 6))
sns.heatmap(xg_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('XGBoost Confusion Matrix')
plt.show()
