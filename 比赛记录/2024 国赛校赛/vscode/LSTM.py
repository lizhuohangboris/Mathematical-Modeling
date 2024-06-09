import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load data
training_data_path = 'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问训练数据.xlsx'
prediction_data_path = 'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问.xlsx'

training_data = pd.read_excel(training_data_path)
prediction_data = pd.read_excel(prediction_data_path)

# Set index to '年份'
training_data.set_index('年份', inplace=True)
prediction_data.set_index('年份', inplace=True)

# Select features and target
features = ['GDP（十亿）', '人口（百万人）', '钢铁产量（千吨）', '水泥（百万吨）', '民用汽车数量（千辆）', '煤炭消耗量（百万吨）', '原油消耗量', '天然气消耗量', '新能源消耗量']
target = '二氧化碳排放量（百万吨）'

# Combine features and target for scaling
training_data_combined = training_data[features + [target]]
prediction_data_combined = prediction_data[features]

# Normalize the features
feature_scaler = MinMaxScaler()
scaled_training_features = feature_scaler.fit_transform(training_data[features])
scaled_prediction_features = feature_scaler.transform(prediction_data[features])

# Normalize the target
target_scaler = MinMaxScaler()
scaled_training_target = target_scaler.fit_transform(training_data[[target]])

# Prepare data for LSTM
def create_dataset(features, target, time_step=1):
    X, Y = [], []
    for i in range(len(features)-time_step-1):
        a = features[i:(i+time_step)]
        X.append(a)
        Y.append(target[i + time_step])
    return np.array(X), np.array(Y)

time_step = 5
X_train, y_train = create_dataset(scaled_training_features, scaled_training_target, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], len(features))

# Prepare prediction data
X_predict = []
for i in range(len(scaled_prediction_features) - time_step):
    X_predict.append(scaled_prediction_features[i:(i + time_step)])
X_predict = np.array(X_predict)
# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, len(features))))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
# Make predictions
train_predict = model.predict(X_train)
future_predict = model.predict(X_predict)

# Inverse transform predictions
train_predict = target_scaler.inverse_transform(train_predict)
future_predict = target_scaler.inverse_transform(future_predict)

# Add predictions to prediction_data
prediction_data['二氧化碳排放量（百万吨）'] = np.nan
prediction_data.iloc[-len(future_predict):, prediction_data.columns.get_loc('二氧化碳排放量（百万吨）')] = future_predict.flatten()

# Save the results to a new Excel file
output_path_lstm = 'D://Normal_tools/Github_desktop/Clone_shop/Mathematical-Modeling/比赛记录/2024 国赛校赛/第二问预测结果_LSTM.xlsx'
prediction_data.to_excel(output_path_lstm, index=False)

output_path_lstm
