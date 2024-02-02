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

# Train Random Forest model on the first 70% of data
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(train[features], train[target])

# Use Random Forest model for predictions on the entire dataset
df['rf_predictions'] = rf_model.predict_proba(df[features])[:, 1]

# Train XGBoost model on the first 70% of data
xg_model = XGBClassifier(random_state=42)
xg_model.fit(train[features], train[target])

# Use XGBoost model for predictions on the entire dataset
df['xg_predictions'] = xg_model.predict_proba(df[features])[:, 1]

# Inverse transform 'point_victor' for interpretation
df['point_victor'] = le.inverse_transform(df['point_victor'])

# Sort DataFrame by 'elapsed_time' for time series plot
df.sort_values('elapsed_time', inplace=True)

# Plot the time series of predicted probabilities for Random Forest
plt.figure(figsize=(10, 6))
plt.plot(df['elapsed_time'], df['rf_predictions'], label='RF Predicted Probability', linestyle='dashed', color='skyblue')

# Plot the time series of predicted probabilities for XGBoost
plt.plot(df['elapsed_time'], df['xg_predictions'], label='XG Predicted Probability', linestyle='dashed', color='salmon')

plt.xlabel('Elapsed Time')
plt.ylabel('Predicted Probability')
plt.title('Time Series of Predicted Probability')
plt.legend()
plt.show()
