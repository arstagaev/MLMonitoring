import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Load reference and new logs
reference_log = pd.read_csv("data/plain1.csv")
new_log = pd.read_csv("data/plain1_anomaly5_3.impulses.csv")

# Ensure time columns are in datetime format
reference_log['time'] = pd.to_datetime(reference_log['time'])
new_log['time'] = pd.to_datetime(new_log['time'])

# Set the parameters for pattern detection
window_size = 50  # Sliding window size for pattern comparison

# Function to calculate the similarity score between two windows
def calculate_similarity(reference_window, new_window):
    # Use Euclidean distance as the similarity measure
    return euclidean(reference_window, new_window)

# Normalize the values of reference and new logs for better comparison
reference_log['value_normalized'] = (reference_log['value'] - reference_log['value'].mean()) / reference_log['value'].std()
new_log['value_normalized'] = (new_log['value'] - new_log['value'].mean()) / new_log['value'].std()

# Prepare rolling windows for reference and new logs
reference_values = reference_log['value_normalized'].values
new_values = new_log['value_normalized'].values
similarity_scores = []

# Calculate similarity scores for each window
for i in range(len(new_values) - window_size + 1):
    reference_window = reference_values[i:i + window_size]
    new_window = new_values[i:i + window_size]
    score = calculate_similarity(reference_window, new_window)
    similarity_scores.append(score)

# Add similarity scores to the new log DataFrame
new_log['similarity_score'] = np.nan
new_log.loc[window_size - 1:, 'similarity_score'] = similarity_scores

# Set a threshold for anomaly detection based on similarity score
anomaly_threshold = np.percentile(similarity_scores, 95)  # Top 5% as anomalies
new_log['anomaly'] = new_log['similarity_score'] > anomaly_threshold

# Plotting
plt.figure(figsize=(15, 8))

# Plot the reference log
plt.plot(reference_log['time'], reference_log['value_normalized'], label="Reference Log (Normalized)", color='magenta', zorder=3, alpha=0.7)

# Plot the new log
plt.plot(new_log['time'], new_log['value_normalized'], label="New Log (Normalized)", color='blue', zorder=2)

# Highlight anomalies
anomalies = new_log[new_log['anomaly']]
plt.scatter(anomalies['time'], anomalies['value_normalized'], color='orange', label="Anomalies (Pattern-Based)", zorder=4)

# Plot similarity score as a secondary plot
plt.twinx()
plt.plot(new_log['time'], new_log['similarity_score'], label="Similarity Score", color='green', alpha=0.5)
plt.axhline(y=anomaly_threshold, color='red', linestyle='--', label="Anomaly Threshold (Similarity)", alpha=0.7)

# Add labels, legend, and grid
plt.xlabel("Time")
plt.ylabel("Value (Normalized)")
plt.title("Pattern-Based Anomaly Detection with Reference Logs")
plt.legend(loc="upper left")
plt.grid(True)

# Show the plot
plt.show()
