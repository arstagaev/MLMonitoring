import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

############################################
# 1. PARAMETERS
############################################
TELEMETRY_FILE = "data_syntetic/1peak.csv"
PEAK_FILE = "data_syntetic/peak_pattern.csv"  # We'll load it, but not embed in this script.

############################################
# 2. LOAD CSV FUNCTION
############################################
def load_csv(filename):
    """
    Reads CSV with columns [time, value].
    Returns (time_array, value_array) sorted by time.
    """
    df = pd.read_csv(filename)
    # Convert columns to numeric
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.dropna(subset=["time","value"], inplace=True)

    # Sort by time to ensure the plot is sequential
    df.sort_values("time", inplace=True)
    
    time = df["time"].values
    values = df["value"].values
    return time, values

############################################
# 3. LOAD TELEMETRY & PATTERN
############################################
time_data, value_data = load_csv(TELEMETRY_FILE)
print(f"Telemetry loaded. Length: {len(value_data)}")

# Optionally load the pattern (not used for embedding here)
pattern_time, pattern_values = load_csv(PEAK_FILE)
print(f"3-peak pattern loaded. Length: {len(pattern_values)}")

############################################
# 4. RUN ISOLATION FOREST ON TELEMETRY
############################################
# We treat each data point as a 1D feature = (value,).
X = value_data.reshape(-1, 1)

iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso.fit(X)
labels = iso.predict(X)  # +1 => normal, -1 => anomaly
anomaly_indices = np.where(labels == -1)[0]
print(f"Detected anomalies: {len(anomaly_indices)}")

############################################
# 5. PLOT TELEMETRY & MARK ANOMALIES
############################################
plt.figure(figsize=(10, 5))
plt.title("3peak Telemetry - Isolation Forest Anomalies")

# Plot data as a line with markers so you see shape changes
plt.plot(time_data, value_data, '-o', color='blue', label='Telemetry')



plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
# Mark anomalies in red
plt.scatter(
    time_data[anomaly_indices],
    value_data[anomaly_indices],
    color='red', s=150, label='Anomaly'
)
plt.show()
