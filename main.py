# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Global variable for time steps used in the sequences
time_steps = 24

# Load and preprocess data function
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    data = df['energy_consumption'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler

# Create sequences from data
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Forecasting function
def forecast(model, recent_data, scaler, time_steps):
    recent_data_scaled = scaler.transform(recent_data.reshape(-1, 1))
    X_recent = recent_data_scaled.reshape(1, time_steps, 1)
    prediction_scaled = model.predict(X_recent)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction

# Main program
if __name__ == "__main__":
    # Load and preprocess historical data
    data_normalized, scaler = load_and_preprocess_data('data.csv')
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

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=72, validation_split=0.2, verbose=1)

    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss: {test_loss}")

    # Load the most recent data for forecasting
    df = pd.read_csv('data.csv')
    recent_data = df['energy_consumption'].values[-time_steps:].reshape(-1, 1)

    # Forecast future energy consumption
    forecasted_consumption = forecast(model, recent_data, scaler, time_steps)
    print(f"Forecasted Energy Consumption: {forecasted_consumption[0][0]}")
