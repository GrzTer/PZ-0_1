import os
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

time_steps = 24


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    data = df['energy_consumption'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler, df


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


class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.history = {'loss': [], 'val_loss': []}

    def on_epoch_end(self, epoch, logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))

        plt.clf()  # Clear the current figure.
        plt.plot(self.history['loss'], label='Training loss')
        plt.plot(self.history['val_loss'], label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.pause(0.1)  # Pause to update the plot


# Load and preprocess historical data
data_normalized, scaler, df = load_and_preprocess_data('data.csv')
X, y = create_sequences(data_normalized, time_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and compile the LSTM model
model = Sequential([
    LSTM(100, activation='relu', return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.3),
    LSTM(50, activation='relu'),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Define callbacks for learning rate adjustment and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)


plot_losses = PlotLosses()

# Define a checkpoint callback
checkpoint_path = "model_checkpoint.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Check if a previous checkpoint exists
if os.path.exists(checkpoint_path):
    print(f"Loading weights from {checkpoint_path}")
    model.load_weights(checkpoint_path)

# Modify the model.fit call to include the checkpoint callback
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=64,  # Adjusted batch size
    validation_split=0.2,
    verbose=1,
    callbacks=[plot_losses, checkpoint, reduce_lr, early_stopping]
)

# Train the model with a validation split and the custom plotting callback
history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_split=0.2, verbose=1,
                    callbacks=[plot_losses])

# Save training history to JSON
with open('training_history.json', 'w') as f:
    json.dump(history.history, f, indent=4)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test, verbose=1)
print(f"Test loss: {test_loss}")

# Forecast future energy consumption
recent_data = df['energy_consumption'].values[-time_steps:].reshape(-1, 1)
forecasted_consumption = forecast(model, recent_data, scaler, time_steps)

# Save forecast to JSON
forecast_result = {'Forecasted Energy Consumption': forecasted_consumption[0][0].tolist()}
with open('forecast_results.json', 'w') as f:
    json.dump(forecast_result, f, indent=4)

print(f"Forecasted Energy Consumption: {forecasted_consumption[0][0]}")
