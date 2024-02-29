import pandas as pd
import numpy as np

# Settings for dummy data generation
n_days = 365  # For one year
hours_per_day = 24
total_hours = n_days * hours_per_day

# Time series
time_series = pd.date_range(start="2020-01-01", periods=total_hours, freq="H")

# Simulate daily patterns and trends
np.random.seed(42)  # For reproducibility
daily_pattern = np.sin(np.arange(hours_per_day) * (2 * np.pi / hours_per_day))
trend = np.linspace(50, 150, total_hours)  # Linear trend from 50 to 150

# Add random noise
noise = np.random.normal(loc=0, scale=5, size=total_hours)

# Simulate energy consumption
energy_consumption = 100 + 25 * daily_pattern.repeat(n_days) + trend + noise

# Create DataFrame
df = pd.DataFrame({
    'timestamp': time_series,
    'energy_consumption': energy_consumption
})

# Save to CSV
df.to_csv('data.csv', index=False)

print("Dummy data generated and saved to 'data.csv'.")
