# Select independent and dependent variables
independent_vars = ['server', 'p1_ace', 'p1_winner', 'p2_winner',
                    'p1_unf_err', 'p1_net_pt_won', 'p2_net_pt_won', 
                    'p1_break_pt', 'p1_break_pt_missed']
dependent_var = 'point_victorr'

# Split the dataset into training and testing sets
X = df[independent_vars]
y = df[dependent_var]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(units=10, input_dim=len(independent_vars), activation='relu'))
model.add(Dense(units=1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=0)

# Predict on the test set
y_pred = model.predict(X_test_scaled)