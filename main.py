import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json

time_steps = 24

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    data = df['energy_consumption'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler, df  # Return the DataFrame as well

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def forecast(model, recent_data, scaler, time_steps):
    recent_data_scaled = scaler.transform(recent_data.reshape(-1, 1))
    X_recent = recent_data_scaled.reshape(1, time_steps, 1)
    prediction_scaled = model.predict(X_recent)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction

# Load and preprocess historical data
data_normalized, scaler, df = load_and_preprocess_data('data.csv')
X, y = create_sequences(data_normalized, time_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and compile the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model with a validation split
history = model.fit(X_train, y_train, epochs=20, batch_size=72, validation_split=0.2, verbose=1)

# Save training history to JSON
with open('training_history.json', 'w') as f:
    json.dump(history.history, f, indent=4)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test, verbose=1)
print(f"Test loss: {test_loss}")

# Load the most recent data for forecasting
recent_data = df['energy_consumption'].values[-time_steps:].reshape(-1, 1)

# Forecast future energy consumption
forecasted_consumption = forecast(model, recent_data, scaler, time_steps)

# Save forecast to JSON
forecast_result = {'Forecasted Energy Consumption': forecasted_consumption[0][0].tolist()}
with open('forecast_results.json', 'w') as f:
    json.dump(forecast_result, f, indent=4)

print(f"Forecasted Energy Consumption: {forecasted_consumption[0][0]}")
