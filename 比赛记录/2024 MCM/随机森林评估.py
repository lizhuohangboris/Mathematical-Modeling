import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

# Standardize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Split data into training and testing sets
train_size = 0.7
train, test = train_test_split(df, test_size=1 - train_size, random_state=42)

# Train Random Forest model on the first 70% of data
rf_model = RandomForestClassifier(random_state=42)

# Define hyperparameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search with 5-fold cross-validation
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(train[features], train[target])

# Get the best parameters
best_params_rf = grid_search_rf.best_params_

# Use the best Random Forest model for predictions on the entire dataset
df['rf_predictions'] = grid_search_rf.predict_proba(df[features])[:, 1]

# Train XGBoost model on the first 70% of data
xg_model = XGBClassifier(random_state=42)

# Define hyperparameter grid for XGBoost
param_grid_xg = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Perform Grid Search with 5-fold cross-validation
grid_search_xg = GridSearchCV(xg_model, param_grid_xg, cv=5, scoring='accuracy')
grid_search_xg.fit(train[features], train[target])

# Get the best parameters
best_params_xg = grid_search_xg.best_params_

# Use the best XGBoost model for predictions on the entire dataset
df['xg_predictions'] = grid_search_xg.predict_proba(df[features])[:, 1]

# Inverse transform 'point_victor' for interpretation
df['point_victor'] = le.inverse_transform(df['point_victor'])

# Model Evaluation
# Assuming 'point_victor' in the original scale is available, if not, use the predicted probabilities directly

# Convert predicted probabilities to binary predictions
df['rf_binary_predictions'] = (df['rf_predictions'] > 0.5).astype(int)
df['xg_binary_predictions'] = (df['xg_predictions'] > 0.5).astype(int)

# Evaluate Random Forest model
print("Random Forest Model Evaluation:")
print("Accuracy:", accuracy_score(df['point_victor'], df['rf_binary_predictions']))
print("Precision:", precision_score(df['point_victor'], df['rf_binary_predictions'], average='micro'))
print("Recall:", recall_score(df['point_victor'], df['rf_binary_predictions'], average='micro'))
print("F1 Score:", f1_score(df['point_victor'], df['rf_binary_predictions'], average='micro'))

# Evaluate XGBoost model
print("\nXGBoost Model Evaluation:")
print("Accuracy:", accuracy_score(df['point_victor'], df['xg_binary_predictions']))
print("Precision:", precision_score(df['point_victor'], df['xg_binary_predictions'], average='micro'))
print("Recall:", recall_score(df['point_victor'], df['xg_binary_predictions'], average='micro'))
print("F1 Score:", f1_score(df['point_victor'], df['xg_binary_predictions'], average='micro'))

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
