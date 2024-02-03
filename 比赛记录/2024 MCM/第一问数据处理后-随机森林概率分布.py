import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'c:/Users/92579/Documents/GitHub/Mathematical-Modeling/比赛记录/2024 MCM/美赛/2024_MCM-ICM_Problems/2024_MCM-ICM_Problems/数据处理.csv'
data = pd.read_csv(file_path)

# Convert 'elapsed_time' to seconds
data['elapsed_time_seconds'] = pd.to_timedelta(data['elapsed_time']).dt.total_seconds()

# Select features and target variable
features = data.drop(['match_id', 'player1', 'player2', 'point_victor', 'elapsed_time'], axis=1)
target = data['point_victor']

# Convert target variable to player 1 scoring
target_player1 = (target == 1).astype(int)

# Split the dataset
train_features, test_features, train_target, test_target = train_test_split(features, target_player1, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(train_features, train_target)

# Predict on the test set
predicted_probabilities = model.predict_proba(test_features)[:, 1]

# Plot the probability distribution
plt.figure(figsize=(10, 6))
sns.histplot(predicted_probabilities, kde=True, bins=50, color='blue', stat='probability')
plt.title('Probability Distribution of Predicted Player 1 Scoring')
plt.xlabel('Predicted Probability for Player 1 Scoring')
plt.ylabel('Probability Density')
plt.show()
