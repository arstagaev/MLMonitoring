import pandas as pd
import numpy as np
import keras
from keras import layers, models
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# Load reference and new logs
reference_log = pd.read_csv("data/plain1.csv")
new_log = pd.read_csv("data/plain1_anomaly5_3.impulses.csv")

# Ensure time columns are in datetime format
reference_log['time'] = pd.to_datetime(reference_log['time'])
new_log['time'] = pd.to_datetime(new_log['time'])

# Parameters for anomaly detection
window_size = 50  # Rolling window size for threshold calculation
threshold_multiplier = 3  # Multiplier for standard deviation to set the anomaly threshold

# Calculate rolling statistics for the reference log
reference_log['rolling_mean'] = reference_log['value'].rolling(window=window_size, center=True).mean()
reference_log['rolling_std'] = reference_log['value'].rolling(window=window_size, center=True).std()

# Define thresholds from reference log
reference_log['upper_threshold'] = reference_log['rolling_mean'] + threshold_multiplier * reference_log['rolling_std']
reference_log['lower_threshold'] = reference_log['rolling_mean'] - threshold_multiplier * reference_log['rolling_std']

# Apply thresholds to the new log
new_log['upper_threshold'] = np.interp(
    new_log['time'].map(pd.Timestamp.timestamp), 
    reference_log['time'].map(pd.Timestamp.timestamp), 
    reference_log['upper_threshold']
)
new_log['lower_threshold'] = np.interp(
    new_log['time'].map(pd.Timestamp.timestamp), 
    reference_log['time'].map(pd.Timestamp.timestamp), 
    reference_log['lower_threshold']
)

# Identify anomalies in the new log
new_log['anomaly'] = (new_log['value'] > new_log['upper_threshold']) | (new_log['value'] < new_log['lower_threshold'])

# Plotting
plt.figure(figsize=(15, 6))

# Plot reference log for comparison
plt.plot(reference_log['time'], reference_log['value'], label="Reference Log", color='magenta', alpha=0.7)

# Plot new log
plt.plot(new_log['time'], new_log['value'], label="New Log", color='blue')
# Overlay reference log
plt.plot(reference_log['time'], reference_log['value'], label="Reference Log", color='magenta', zorder=3, alpha=0.7)

# Plot thresholds
plt.plot(new_log['time'], new_log['upper_threshold'], label="Upper Threshold", color='green', linestyle='--')
plt.plot(new_log['time'], new_log['lower_threshold'], label="Lower Threshold", color='red', linestyle='--')

# Highlight anomalies in the new log
anomalies = new_log[new_log['anomaly']]
plt.scatter(anomalies['time'], anomalies['value'], color='orange', label="Anomalies", zorder=5)

# Add labels, legend, and grid
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Anomaly Detection Based on Reference Log")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

