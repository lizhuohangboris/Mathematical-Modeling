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
columns_to_read = ['elapsed_time', 'game_victor', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'score_lead', 'Tie_breakers', 'server', 'serve_no']

df = pd.read_csv(file_path, usecols=columns_to_read)

# Convert 'elapsed_time' to datetime
df['elapsed_time'] = pd.to_datetime(df['elapsed_time'])

# Choose features and target variable
features_rf = ['p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'score_lead']
features_xg = features_rf + ['rf_predictions']  # Include Random Forest predictions as a feature for XGBoost
target = 'game_victor'

# Encode 'game_victor' column
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

# Inverse transform 'game_victor' for interpretation
df['game_victor'] = le.inverse_transform(df['game_victor'])

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

# Plot the time series of predicted probabilities for XGBoost
plt.figure(figsize=(15, 8))

# Subplot 1: XGBoost Predicted Probability and Original Data Points
plt.plot(df['elapsed_time'], xg_model.predict_proba(df[features_xg])[:, 1], label='XGBoost Predicted Probability (Player 1)', linestyle='dashed', color='salmon')
plt.plot(df['elapsed_time'], xg_model.predict_proba(df[features_xg])[:, 0], label='XGBoost Predicted Probability (Player 2)', linestyle='dashed', color='green')
# plt.scatter(df['elapsed_time'], df['game_victor'], marker='o', s=5, color='black', label='Original Data Points - XG')
plt.scatter(df['elapsed_time'], xg_model.predict_proba(df[features_xg])[:, 1], marker='o', s=5, color='blue', label='Predicted Points - XG')  # Use the correct column
plt.xlabel('Elapsed Time')
plt.ylabel('Predicted Probability / Game Victor (0 or 1)')
plt.title('XGBoost - Time Series of Predicted Probability with Scatter Plot')
plt.legend()

plt.tight_layout()
plt.show()

# Check unique values of 'game_victor' after encoding
print(df['game_victor'].unique())
