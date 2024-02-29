import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_daily_data(start_date, days=365, filename='data.csv'):
    # Settings for dummy data generation
    n_days = days  # Generate data for specified days
    hours_per_day = 24
    total_hours = n_days * hours_per_day

    # Determine the start date for the new data
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        last_date = pd.to_datetime(df['timestamp'].iloc[-1])
        start_date = last_date + timedelta(hours=1)
    else:
        start_date = pd.to_datetime(start_date)

    # Time series for specified days
    time_series = pd.date_range(start=start_date, periods=total_hours, freq="H")

    # Ensure reproducibility
    np.random.seed(42)

    # Simulate daily patterns and trends over multiple days
    daily_pattern = np.tile(np.sin(np.arange(hours_per_day) * (2 * np.pi / hours_per_day)), n_days)

    # Assuming trend continues from the last value if data exists
    if os.path.exists(filename):
        trend_start = df['energy_consumption'].iloc[-1]
    else:
        trend_start = 100  # Starting point of the trend for the new data
    trend = np.linspace(trend_start, trend_start + n_days * 2, total_hours)  # Gradual trend increase over days

    # Add random noise
    noise = np.random.normal(loc=0, scale=5, size=total_hours)

    # Simulate energy consumption for specified days
    energy_consumption = trend + 25 * daily_pattern + noise

    # Create DataFrame
    daily_df = pd.DataFrame({
        'timestamp': time_series,
        'energy_consumption': energy_consumption
    })

    # Append or save to CSV
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = pd.concat([df, daily_df], ignore_index=True)
        df.to_csv(filename, index=False)
    else:
        daily_df.to_csv(filename, index=False)

    print(f"Data for {n_days} days generated and appended to '{filename}'.")

# Example usage
generate_daily_data("2020-01-01", days=30)  # Adjust 'days' parameter as needed
