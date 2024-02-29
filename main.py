import os
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Set seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

time_steps = 24  # Adjusted time steps

# Load and preprocess data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    data = df['energy_consumption'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler, df

# Create sequences for LSTM
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Forecast function
def forecast(model, recent_data, scaler, time_steps):
    recent_data_scaled = scaler.transform(recent_data.reshape(-1, 1))
    X_recent = recent_data_scaled.reshape(1, time_steps, 1)
    prediction_scaled = model.predict(X_recent)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction

data_normalized, scaler, df = load_and_preprocess_data('data.csv')
X, y = create_sequences(data_normalized, time_steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(100, activation='relu', return_sequences=True, input_shape=(time_steps, 1)),
    Dropout(0.3),
    LSTM(50, activation='relu'),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
plot_losses = PlotLosses()
checkpoint_path = "model_checkpoint.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')

if os.path.exists(checkpoint_path):
    try:
        model.load_weights(checkpoint_path)
    except Exception as e:
        print(f"Error loading weights: {e}")

history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_split=0.2, callbacks=[reduce_lr, early_stopping, plot_losses, checkpoint], verbose=1)

# Function to convert numpy.float32 to Python float for JSON serialization
def convert(o):
    if isinstance(o, np.float32): return float(o)
    raise TypeError

# Save training history to JSON, using `convert` for serialization
with open('training_history.json', 'w') as f:
    json.dump(history.history, f, indent=4, default=convert)

test_loss = model.evaluate(X_test, y_test, verbose=1)
print(f"Test loss: {test_loss}")

recent_data = df['energy_consumption'].values[-time_steps:].reshape(-1, 1)
forecasted_consumption = forecast(model, recent_data, scaler, time_steps)

# Convert forecasted consumption to float before saving
forecast_result = {'Forecasted Energy Consumption': float(forecasted_consumption[0][0])}
with open('forecast_results.json', 'w') as f:
    json.dump(forecast_result, f, indent=4)

print(f"Forecasted Energy Consumption: {forecasted_consumption[0][0]}")
