import pandas as pd
from sklearn.model_selection import train_test_split
from prophet import Prophet
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

# Convert 'elapsed_time' to datetime
df['elapsed_time'] = pd.to_datetime(df['elapsed_time'])

# Choose features and target variable
features = ['p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'score_lead']
target = 'point_victor'

# Encode 'point_victor' column
le = LabelEncoder()
df[target] = le.fit_transform(df[target])

# Split data into training and testing sets
train_size = 0.7
train, test = train_test_split(df, test_size=1 - train_size, random_state=42)

# Prophet model for time series prediction
prophet_model = Prophet()
prophet_model.add_regressor('p1_sets')
prophet_model.add_regressor('p2_sets')
prophet_model.add_regressor('p1_games')
prophet_model.add_regressor('p2_games')
prophet_model.add_regressor('score_lead')

# Rename columns for Prophet
train_prophet = train[['elapsed_time', 'point_victor', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'score_lead']]
train_prophet = train_prophet.rename(columns={'elapsed_time': 'ds', 'point_victor': 'y'})

# Fit the Prophet model
prophet_model.fit(train_prophet)

# Create a dataframe with future timestamps for prediction
future = pd.DataFrame(pd.date_range(start=df['elapsed_time'].min(), end=df['elapsed_time'].max(), freq='1H'), columns=['ds'])

# Add regressor values for the future dataframe
for feature in features:
    future[feature] = df[feature].mean()  # Use mean value for regressors in the future

# Make predictions on the entire dataset
prophet_predictions = prophet_model.predict(future)

# Merge Prophet predictions with the original DataFrame
df = pd.merge(df, prophet_predictions[['ds', 'yhat']], how='left', left_on='elapsed_time', right_on='ds')

# Fill NaN values in 'yhat' with 0
df['yhat'] = df['yhat'].fillna(0)

# Add 'yhat' as a feature for XGBoost
features.append('yhat')

# Train XGBoost model on the entire dataset
xg_model = XGBClassifier(random_state=42)
xg_model.fit(df[features], df[target])



# Use XGBoost model for predictions on the entire dataset
df['xg_predictions'] = xg_model.predict(df[features])

# Inverse transform 'point_victor' for interpretation
df['point_victor'] = le.inverse_transform(df['point_victor'])

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
plt.plot(df['elapsed_time'], xg_model.predict_proba(df[features])[:, 1], label='XGBoost Predicted Probability (Player 1)', linestyle='dashed', color='salmon')
plt.plot(df['elapsed_time'], xg_model.predict_proba(df[features])[:, 0], label='XGBoost Predicted Probability (Player 2)', linestyle='dashed', color='green')
plt.scatter(df['elapsed_time'], df['point_victor'], marker='o', s=5, color='black', label='Original Data Points - XG')
plt.scatter(df['elapsed_time'], xg_model.predict_proba(df[features])[:, 1], marker='o', s=5, color='blue', label='Predicted Points - XG')  # Blue points at predicted points
plt.xlabel('Elapsed Time')
plt.ylabel('Predicted Probability / Point Victor (0 or 1)')
plt.title('XGBoost - Time Series of Predicted Probability with Scatter Plot')
plt.legend()

plt.tight_layout()
plt.show()
