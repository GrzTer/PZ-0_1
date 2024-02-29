import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt

time_steps = 24


class TrainingVisualization(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        labels = ['Training Loss', 'Validation Loss']
        sizes = [logs['loss'], logs['val_loss']]  # Use loss values from logs
        colors = ['skyblue', 'yellowgreen']

        plt.figure(figsize=(5, 5))  # Set figure size
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
        plt.title(f'Loss Distribution at Epoch {epoch + 1}')
        plt.show()


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


data_normalized, scaler, df = load_and_preprocess_data('data.csv')
X, y = create_sequences(data_normalized, time_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model with a validation split and custom callback
vis_callback = TrainingVisualization()
history = model.fit(X_train, y_train, epochs=1000, batch_size=72, validation_split=0.2, verbose=1,
                    callbacks=[vis_callback])

# Save training history to JSON
with open('training_history.json', 'w') as f:
    json.dump(history.history, f, indent=4)

test_loss = model.evaluate(X_test, y_test, verbose=1)
print(f"Test loss: {test_loss}")

recent_data = df['energy_consumption'].values[-time_steps:].reshape(-1, 1)
forecasted_consumption = forecast(model, recent_data, scaler, time_steps)

forecast_result = {'Forecasted Energy Consumption': forecasted_consumption[0][0].tolist()}
with open('forecast_results.json', 'w') as f:
    json.dump(forecast_result, f, indent=4)

print(f"Forecasted Energy Consumption: {forecasted_consumption[0][0]}")
