import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
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
features = ['elapsed_time', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'score_lead']
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

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(train[features], train[target])

# Train Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(train[features], train[target])

# Train Bagging model with Random Forest as base estimator
bagging_model = BaggingClassifier(base_estimator=RandomForestClassifier(random_state=42), random_state=42)
bagging_model.fit(train[features], train[target])

# Make predictions on the test dataset
test['rf_predictions'] = rf_model.predict(test[features])
test['gb_predictions'] = gb_model.predict(test[features])
test['bagging_predictions'] = bagging_model.predict(test[features])

# Calculate accuracy for Random Forest
rf_accuracy = accuracy_score(test[target], test['rf_predictions'])
print("Random Forest Accuracy:", rf_accuracy)

# Calculate accuracy for Gradient Boosting
gb_accuracy = accuracy_score(test[target], test['gb_predictions'])
print("Gradient Boosting Accuracy:", gb_accuracy)

# Calculate accuracy for Bagging
bagging_accuracy = accuracy_score(test[target], test['bagging_predictions'])
print("Bagging Accuracy:", bagging_accuracy)

# Confusion matrix for Random Forest
rf_conf_matrix = confusion_matrix(test[target], test['rf_predictions'])
plt.figure(figsize=(8, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Confusion matrix for Gradient Boosting
gb_conf_matrix = confusion_matrix(test[target], test['gb_predictions'])
plt.figure(figsize=(8, 6))
sns.heatmap(gb_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Gradient Boosting Confusion Matrix')
plt.show()

# Confusion matrix for Bagging
bagging_conf_matrix = confusion_matrix(test[target], test['bagging_predictions'])
plt.figure(figsize=(8, 6))
sns.heatmap(bagging_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Bagging Confusion Matrix')
plt.show()
