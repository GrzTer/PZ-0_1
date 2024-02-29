import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
import joblib
import time
import os

# Constants
time_steps = 24

# Function to load and preprocess data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    data = df['energy_consumption'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler, df

# Function to create sequences
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Function to forecast future energy consumption
def forecast(model, recent_data, scaler, time_steps):
    recent_data_scaled = scaler.transform(recent_data.reshape(-1, 1))
    X_recent = recent_data_scaled.reshape(1, time_steps, 1)
    prediction_scaled = model.predict(X_recent)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction

# Continuous learning loop
def continuous_learning_loop(filepath, model_path, scaler_path, check_interval=3600):
    while True:
        # Check for the existence of the model and scaler, load if exist
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
        else:
            model, scaler = None, None  # Initialize these properly as needed

        data_normalized, scaler, df = load_and_preprocess_data(filepath)
        X, y = create_sequences(data_normalized, time_steps)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model is None:
            # Define and compile a new LSTM model if not loaded
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

        # Train the model with a validation split
        history = model.fit(X_train, y_train, epochs=5, batch_size=72, validation_split=0.2, verbose=1)

        # Visualization
        labels = ['Training Data', 'Validation Data']
        sizes = [len(X_train), len(X_train) * 0.2]  # 20% data is for validation as per validation_split
        colors = ['skyblue', 'yellowgreen']
        explode = (0.1, 0)  # explode the first slice

        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.title('Distribution of Data for Latest Training Cycle')
        plt.show()

        # Save the updated model and scaler
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        # Sleep before the next check
        time.sleep(check_interval)

if __name__ == "__main__":
    continuous_learning_loop('data.csv', 'lstm_model.h5', 'scaler.save')
