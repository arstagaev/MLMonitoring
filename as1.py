import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks

############################################
# 1. PARAMETERS
############################################

PATTERN_CSV = "anomaly_pattern.csv"
TELEMETRY_CSV = "telemetry.csv"

# Threshold for correlation peak detection (adjust to your data)
CORR_THRESHOLD = 5.0  

############################################
# 2. HELPER FUNCTION TO READ CSV
############################################

def load_csv(filename):
    """
    Reads a CSV with columns [time, value] and returns:
    - time: 1D numpy array of time indices
    - values: 1D numpy array of sensor values
    """
    df = pd.read_csv(filename)
    time = df["time"].values
    values = df["value"].values
    return time, values

############################################
# 3. LOAD PATTERN AND TELEMETRY
############################################

# 3.1 Load the known anomaly pattern
pattern_time, pattern_values = load_csv(PATTERN_CSV)
pattern_length = len(pattern_values)
print(f"Loaded anomaly pattern of length {pattern_length}")

# 3.2 Load the main telemetry signal
telemetry_time, telemetry_values = load_csv(TELEMETRY_CSV)
telemetry_length = len(telemetry_values)
print(f"Loaded telemetry signal of length {telemetry_length}")

############################################
# 4. CROSS-CORRELATION
############################################

# Perform cross-correlation
# mode='full' gives the complete correlation array of length (N+M-1).
# mode='same' is also common, but we'll use 'full' here for clarity.
corr = correlate(telemetry_values, pattern_values, mode='full')

# (Optional) Normalize the correlation by the pattern's energy 
pattern_energy = np.sum(pattern_values**2)
corr_norm = corr / np.sqrt(pattern_energy)

# The index range of `corr_norm` is roughly 0..(telemetry_length + pattern_length - 2).

############################################
# 5. FIND SIGNIFICANT PEAKS
############################################

# We'll look for peaks above a correlation threshold (CORR_THRESHOLD).
peaks, properties = find_peaks(corr_norm, height=CORR_THRESHOLD)
peak_heights = properties["peak_heights"]

print("Peaks found in cross-correlation at indices:", peaks)
print("Corresponding peak correlation values:", peak_heights)

# Convert correlation indices to approximate alignment in the telemetry signal.
# For cross-correlation (full), if we call the correlation array index i:
#   the pattern is aligned with the signal around i - (pattern_length - 1)
detected_locs = peaks - (pattern_length - 1)
print("Estimated anomaly start indices in telemetry:", detected_locs)

############################################
# 6. PLOT RESULTS
############################################

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

# 6.1 Plot the telemetry signal
axs[0].plot(telemetry_time, telemetry_values, label="Telemetry Signal", color='b')
axs[0].set_title("Telemetry with Possible Anomaly Locations")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Value")

# Mark on the telemetry plot where we found anomalies
for loc in detected_locs:
    # Ensure the location is within the valid range of the telemetry
    if 0 <= loc < telemetry_length:
        axs[0].axvline(telemetry_time[loc], color='r', linestyle='--')
        axs[0].text(telemetry_time[loc], np.max(telemetry_values), 
                    f"Detected @ {telemetry_time[loc]}", color='red')

axs[0].legend()

# 6.2 Plot the normalized cross-correlation
axs[1].plot(corr_norm, label="Normalized Cross-Correlation", color='g')
axs[1].plot(peaks, corr_norm[peaks], "rx", label="Detected Peaks")
axs[1].axhline(CORR_THRESHOLD, color='orange', linestyle='--', label=f"Threshold={CORR_THRESHOLD}")
axs[1].set_title("Cross-Correlation with the Known Pattern")
axs[1].set_xlabel("Correlation Array Index")
axs[1].set_ylabel("Correlation")
axs[1].legend()

plt.tight_layout()
plt.show()