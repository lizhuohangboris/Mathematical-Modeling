import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from '数据处理.csv' with specific columns
file_path = "c:/Users/92579/Documents/GitHub/Mathematical-Modeling/比赛记录/2024 MCM/美赛/2024_MCM-ICM_Problems/2024_MCM-ICM_Problems/数据处理.csv"
columns_to_read = ['elapsed_time', 'point_victor', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'score_lead', 'Tie_breakers', 'server', 'serve_no']

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

# Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(train[features], train[target])

# Use Random Forest model for predictions on the test set
test['rf_predictions'] = rf_model.predict(test[features])

# Calculate accuracy for Random Forest on the test set
rf_accuracy = accuracy_score(test[target], test['rf_predictions'])
print("Random Forest Accuracy on Test Set:", rf_accuracy)

# Confusion matrix for Random Forest
rf_conf_matrix = confusion_matrix(test[target], test['rf_predictions'])
plt.figure(figsize=(8, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest Confusion Matrix on Test Set')
plt.show()

# Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(train[features], train[target])

# Use Gradient Boosting model for predictions on the test set
test['gb_predictions'] = gb_model.predict(test[features])

# Calculate accuracy for Gradient Boosting on the test set
gb_accuracy = accuracy_score(test[target], test['gb_predictions'])
print("Gradient Boosting Accuracy on Test Set:", gb_accuracy)

# Confusion matrix for Gradient Boosting
gb_conf_matrix = confusion_matrix(test[target], test['gb_predictions'])
plt.figure(figsize=(8, 6))
sns.heatmap(gb_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Gradient Boosting Confusion Matrix on Test Set')
plt.show()
