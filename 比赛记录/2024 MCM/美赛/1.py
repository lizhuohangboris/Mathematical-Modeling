import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load data
file_path = "c:/Users/92579/Documents/GitHub/Mathematical-Modeling/比赛记录/2024 MCM/美赛/2024_MCM-ICM_Problems/2024_MCM-ICM_Problems/de.xlsx"
df = pd.read_excel(file_path)
print(df.columns)

# Choose features and target variable
features = ['elapsed_time', 'set_no', 'game_no', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games','p1_score','p2_score',
            'server_1', 'server_2', 'serve_no_1', 'serve_no_2', 
            'p1_points_won','p2_points_won','game_victor','set_victor','p1_ace','p2_ace','p1_winner','p2_winner',
            'p1_double_fault','p2_double_fault','p1_unf_err','p2_unf_err',
            'p1_net_pt','p2_net_pt','p1_net_pt_won','p2_net_pt_won',
            'p1_break_pt','p2_break_pt','p1_break_pt_won','p2_break_pt_won','p1_break_pt_missed','p2_break_pt_missed',
            'p1_distance_run','p2_distance_run','rally_count','speed_mph']


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

# Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(train[features], train[target])

# Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(train[features], train[target])

# Predict on the test set
test['predicted_prob_rf'] = rf_model.predict_proba(test[features])[:, 1]
test['predicted_prob_gb'] = gb_model.predict_proba(test[features])[:, 1]

# Inverse transform 'point_victor' for interpretation
test['point_victor'] = le.inverse_transform(test['point_victor'])

# Sort DataFrame by 'elapsed_time' for time series plot
test.sort_values('elapsed_time', inplace=True)

# Plot the time series of predicted probabilities
plt.figure(figsize=(10, 6))
plt.plot(test['elapsed_time'], test['predicted_prob_rf'], label='RF Predicted Probability', linestyle='dashed')
plt.plot(test['elapsed_time'], test['predicted_prob_gb'], label='GB Predicted Probability', linestyle='dashed')
plt.xlabel('Elapsed Time')
plt.ylabel('Predicted Probability')
plt.title('Time Series of Predicted Probability')
plt.legend()
plt.show()

# Predict and evaluate
rf_predictions = rf_model.predict(test[features])
gb_predictions = gb_model.predict(test[features])

# Inverse transform 'point_victor' column
test['predicted_rf'] = le.inverse_transform(rf_predictions)
test['predicted_gb'] = le.inverse_transform(gb_predictions)

# Calculate accuracy
accuracy_rf = accuracy_score(test['point_victor'], test['predicted_rf'])
accuracy_gb = accuracy_score(test['point_victor'], test['predicted_gb'])

print(f'Random Forest Accuracy: {accuracy_rf:.4f}')
print(f'Gradient Boosting Accuracy: {accuracy_gb:.4f}')
