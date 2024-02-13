import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

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

# Train Random Forest model on the first 70% of data
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(train[features], train[target])

# Use Random Forest model for predictions on the test set
rf_predictions = rf_model.predict(test[features])
rf_probs = rf_model.predict_proba(test[features])[:, 1]

# Train XGBoost model on the first 70% of data without 'score_lead'
xg_model = XGBClassifier(random_state=42)
xg_model.fit(train[features], train[target])

# Use XGBoost model for predictions on the test set
xg_predictions = xg_model.predict(test[features])
xg_probs = xg_model.predict_proba(test[features])[:, 1]

# ... (rest of the code)


# Evaluate Random Forest model
rf_accuracy = accuracy_score(test[target], rf_predictions)
rf_conf_matrix = confusion_matrix(test[target], rf_predictions)
rf_fpr, rf_tpr, _ = roc_curve(test[target], rf_probs)
rf_auc = auc(rf_fpr, rf_tpr)

# Evaluate XGBoost model
xg_accuracy = accuracy_score(test[target], xg_predictions)
xg_conf_matrix = confusion_matrix(test[target], xg_predictions)
xg_fpr, xg_tpr, _ = roc_curve(test[target], xg_probs)
xg_auc = auc(xg_fpr, xg_tpr)

# Plot ROC curves
plt.figure(figsize=(10, 6))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})', color='skyblue')
plt.plot(xg_fpr, xg_tpr, label=f'XGBoost (AUC = {xg_auc:.2f})', color='salmon')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Plot Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrices', fontsize=16)

# Random Forest
axes[0].imshow(rf_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
axes[0].set_title('Random Forest Confusion Matrix')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['0', '1'])
axes[0].set_yticklabels(['0', '1'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# XGBoost
axes[1].imshow(xg_conf_matrix, interpolation='nearest', cmap=plt.cm.Reds)
axes[1].set_title('XGBoost Confusion Matrix')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['0', '1'])
axes[1].set_yticklabels(['0', '1'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.show()


# ... （之前的代码）

# Evaluate Random Forest model
rf_accuracy = accuracy_score(test[target], rf_predictions)
rf_conf_matrix = confusion_matrix(test[target], rf_predictions)
rf_fpr, rf_tpr, _ = roc_curve(test[target], rf_probs)
rf_auc = auc(rf_fpr, rf_tpr)

print("Random Forest Model Evaluation:")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Confusion Matrix:\n{rf_conf_matrix}")
print(f"AUC: {rf_auc:.4f}")

# Evaluate XGBoost model
xg_accuracy = accuracy_score(test[target], xg_predictions)
xg_conf_matrix = confusion_matrix(test[target], xg_predictions)
xg_fpr, xg_tpr, _ = roc_curve(test[target], xg_probs)
xg_auc = auc(xg_fpr, xg_tpr)

print("\nXGBoost Model Evaluation:")
print(f"Accuracy: {xg_accuracy:.4f}")
print(f"Confusion Matrix:\n{xg_conf_matrix}")
print(f"AUC: {xg_auc:.4f}")

# ... （后续的代码）
