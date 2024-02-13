import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load data
file_path = "c:/Users/92579/Documents/GitHub/Mathematical-Modeling/比赛记录/2024 MCM/美赛/2024_MCM-ICM_Problems/2024_MCM-ICM_Problems/de.xlsx"
df = pd.read_excel(file_path)

# Choose features and target variable
features = ['elapsed_time', 'set_no', 'game_no', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games']
target = 'point_victor'

# Convert 'elapsed_time' to string and then to timedelta
df['elapsed_time'] = pd.to_timedelta(df['elapsed_time'].astype(str)).dt.total_seconds()

# Process other non-numeric features (dummy encoding)
df = pd.get_dummies(df, columns=['server', 'serve_no', 'winner_shot_type'])

# Encode 'point_victor' column
le = LabelEncoder()
df[target] = le.fit_transform(df[target])

# Split data into training and testing sets
train_size = 0.7
train, test = train_test_split(df, test_size=1 - train_size, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(train[features], train[target])

# Train XGBoost model
xg_model = XGBClassifier(random_state=42)
xg_model.fit(train[features], train[target])

# Predict on the test set
test['predicted_prob_rf'] = rf_model.predict_proba(test[features])[:, 1]
test['predicted_prob_xg'] = xg_model.predict_proba(test[features])[:, 1]

# Predict on the train set
train['predicted_prob_rf'] = rf_model.predict_proba(train[features])[:, 1]
train['predicted_prob_xg'] = xg_model.predict_proba(train[features])[:, 1]

# Inverse transform 'point_victor' for interpretation
test['point_victor'] = le.inverse_transform(test['point_victor'])
train['point_victor'] = le.inverse_transform(train['point_victor'])

# Sort DataFrame by 'elapsed_time' for time series plot
test.sort_values('elapsed_time', inplace=True)
train.sort_values('elapsed_time', inplace=True)

# Plot the time series of predicted probabilities for Random Forest
plt.figure(figsize=(10, 6))
plt.plot(test['elapsed_time'], test['predicted_prob_rf'], label='RF Test Predicted Probability', linestyle='dashed', color='blue')
plt.xlabel('Elapsed Time')
plt.ylabel('Predicted Probability')
plt.title('Time Series of Predicted Probability (Random Forest)')
plt.legend()
plt.show()

# Plot the time series of predicted probabilities for XGBoost
plt.figure(figsize=(10, 6))
plt.plot(test['elapsed_time'], test['predicted_prob_xg'], label='XG Test Predicted Probability', linestyle='dashed', color='red')
plt.xlabel('Elapsed Time')
plt.ylabel('Predicted Probability')
plt.title('Time Series of Predicted Probability (XGBoost)')
plt.legend()
plt.show()
