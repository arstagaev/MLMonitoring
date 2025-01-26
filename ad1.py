import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

############################################
# 1. PARAMETERS
############################################

# Filenames (adjust to your actual file names)
# NORMAL_CSV = "normal_vibration.csv"
# ANOMALY_CSV = "anomaly_vibration.csv"
NORMAL_CSV = "data/plain1.csv"
ANOMALY_CSV = "data/plain1_anomaly2.csv"

# Windowing parameters
WINDOW_SIZE = 50   # how many consecutive points form one sequence
STEP_SIZE = 1      # slide the window by 1 each time

# Autoencoder hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LATENT_DIM = 8     # dimension of the bottleneck in the autoencoder

############################################
# 2. HELPER FUNCTIONS
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

def create_windows(values, window_size=50, step=1):
    """
    Splits the 1D 'values' array into overlapping windows of size 'window_size'.
    Moves 'step' points each time. Returns a 2D array of shape (num_windows, window_size).
    Also returns an array of the starting indices (for plotting).
    """
    windows = []
    indices = []
    i = 0
    while (i + window_size) <= len(values):
        window = values[i : i + window_size]
        windows.append(window)
        indices.append(i)
        i += step
    
    return np.array(windows), np.array(indices)

############################################
# 3. LOAD AND PREPARE NORMAL DATA
############################################

# Load normal reference data
time_normal, values_normal = load_csv(NORMAL_CSV)
print(f"Normal data shape: {values_normal.shape}")

# Create overlapping windows from the normal data
normal_windows, normal_indices = create_windows(values_normal, 
                                               window_size=WINDOW_SIZE, 
                                               step=STEP_SIZE)

# For neural networks, we often expand dims to (num_samples, window_size, 1)
normal_windows_reshaped = np.expand_dims(normal_windows, axis=-1)

############################################
# 4. BUILD THE AUTOENCODER MODEL
############################################

# A simple 1D autoencoder:
# - Encoder: 1D Conv layers or Dense flatten approach
# - Bottleneck (latent dimension)
# - Decoder: tries to reconstruct original window

# Option A: Flatten + Dense approach (simpler for demonstration)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(WINDOW_SIZE, 1)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(LATENT_DIM, activation='relu'),  # Bottleneck
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(WINDOW_SIZE, activation='linear'),  # Output dimension same as input
    keras.layers.Reshape((WINDOW_SIZE, 1))
])

model.compile(optimizer='adam', loss='mse')
model.summary()

############################################
# 5. TRAIN THE MODEL ON NORMAL DATA
############################################

history = model.fit(
    normal_windows_reshaped, normal_windows_reshaped,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

############################################
# 6. DETERMINE A RECONSTRUCTION ERROR THRESHOLD
############################################
# We'll get reconstruction errors on the normal training set
# and pick a threshold, e.g. mean + 3*std, or tune as needed

reconstructions = model.predict(normal_windows_reshaped)
mse = np.mean(np.power(
    normal_windows_reshaped - reconstructions, 2), axis=(1,2))

mean_mse = np.mean(mse)
std_mse = np.std(mse)
threshold = mean_mse + 3 * std_mse

print(f"Reconstruction error mean: {mean_mse:.5f}, std: {std_mse:.5f}")
print(f"Selected anomaly threshold: {threshold:.5f}")

############################################
# 7. LOAD ANOMALY DATA AND DETECT ANOMALIES
############################################

time_anomaly, values_anomaly = load_csv(ANOMALY_CSV)
print(f"Anomaly data shape: {values_anomaly.shape}")

# Create windows on the anomaly data
anomaly_windows, anomaly_indices = create_windows(values_anomaly, 
                                                  window_size=WINDOW_SIZE, 
                                                  step=STEP_SIZE)
anomaly_windows_reshaped = np.expand_dims(anomaly_windows, axis=-1)

# Get reconstruction from the model
anomaly_reconstructions = model.predict(anomaly_windows_reshaped)
mse_anomaly = np.mean(np.power(
    anomaly_windows_reshaped - anomaly_reconstructions, 2), axis=(1,2))

# Compare each window's MSE to the threshold
anomaly_flags = mse_anomaly > threshold

############################################
# 8. PLOTTING THE RESULTS
############################################

# We'll create a figure that shows:
# - The original anomaly signal
# - Markers for where we detect anomalies

plt.figure(figsize=(12, 6))
plt.title("Anomaly Detection on Test CSV")

# Plot the entire vibration signal
plt.plot(time_anomaly, values_anomaly, label="Vibration")

# For windows flagged as anomalies, we can highlight the center
for i, flag in enumerate(anomaly_flags):
    if flag:
        # The window starts at anomaly_indices[i], ends at anomaly_indices[i] + WINDOW_SIZE
        start_idx = anomaly_indices[i]
        end_idx = start_idx + WINDOW_SIZE
        
        # We'll mark the middle time step of this window as anomalous
        mid_idx = (start_idx + end_idx) // 2
        plt.axvspan(time_anomaly[start_idx], 
                    time_anomaly[end_idx-1],
                    color='red', alpha=0.2)
        # Alternatively, just plot a point:
        # plt.scatter(time_anomaly[mid_idx], values_anomaly[mid_idx], color='red')

plt.xlabel("Time")
plt.ylabel("Vibration Value")
plt.legend()
plt.show()