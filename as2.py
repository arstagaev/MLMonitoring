import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from datetime import datetime
# Last quite good working prototype

############################################
# 1. PARAMETERS
############################################

ANOMALY_PATTERN_CSV = "data_phone/slam_two_peaks_pattern.csv"
#TELEMETRY_CSV = "data/plain1_anomaly5_3.impulses_short_peak.csv"
TELEMETRY_CSV = "data_phone/two_slams.csv"


WINDOW_SIZE = 50      # Each training window length
POSITIVE_SAMPLES = 100
NEGATIVE_SAMPLES = 200

EPOCHS = 10
BATCH_SIZE = 32

# Get the current date and time
current_datetime = datetime.now()

# Format the date and time
formatted_datetime = current_datetime.strftime("%m%d_%H%M%S")

TFLITE_MODEL_PATH = f"tfmodels/model_{formatted_datetime}.tflite"

# For final detection
ANOMALY_THRESHOLD = 0.75 #0.5  # Probability above this => anomaly
NORMAL_BAND_MULT = 2.0   # Plot ±2 std around mean in yellow

############################################
# 2. HELPER FUNCTIONS
############################################

def load_csv(filename):
    df = pd.read_csv(filename)
    
    # Convert time and value columns to float (if they're valid numeric data)
    df["time"] = pd.to_numeric(df["time"], errors='coerce')
    df["value"] = pd.to_numeric(df["value"], errors='coerce')
    
    # Drop rows that become NaN if there's bad data
    df.dropna(subset=["time", "value"], inplace=True)
    
    time = df["time"].values
    values = df["value"].values
    return time, values

def embed_short_pattern(telemetry, pattern, window_size):
    """
    Creates a window of length 'window_size' from telemetry
    and embeds the entire 'pattern' (23 samples) at a random
    position inside that window.
    """
    N = len(telemetry)
    start_idx = np.random.randint(0, N - window_size)
    window = np.copy(telemetry[start_idx : start_idx + window_size])

    p_len = len(pattern)
    if p_len <= window_size:
        insert_idx = np.random.randint(0, window_size - p_len + 1)
        window[insert_idx:insert_idx + p_len] += pattern
    else:
        # If pattern is bigger than the window, just truncate it for demo
        window = pattern[:window_size]
    return window

def random_window(telemetry, window_size):
    """
    Picks a random window of length 'window_size' from telemetry
    without embedding the pattern.
    """
    N = len(telemetry)
    start_idx = np.random.randint(0, N - window_size)
    return np.copy(telemetry[start_idx : start_idx + window_size])

############################################
# 3. LOAD DATA
############################################

pattern_time, pattern_vals = load_csv(ANOMALY_PATTERN_CSV)  # 23 samples
tele_time, tele_vals       = load_csv(TELEMETRY_CSV)

print(f"Loaded short anomaly pattern: {len(pattern_vals)} samples")
print(f"Loaded telemetry: {len(tele_vals)} samples")

############################################
# 4. CREATE A LABELED DATASET
############################################

# Positive examples (contains short pattern)
X_pos = []
y_pos = []
for _ in range(POSITIVE_SAMPLES):
    w = embed_short_pattern(tele_vals, pattern_vals, WINDOW_SIZE)
    X_pos.append(w)
    y_pos.append(1)

# Negative examples
X_neg = []
y_neg = []
for _ in range(NEGATIVE_SAMPLES):
    w = random_window(tele_vals, WINDOW_SIZE)
    X_neg.append(w)
    y_neg.append(0)

X = np.array(X_pos + X_neg)
y = np.array(y_pos + y_neg)
print("Dataset shape:", X.shape, "Labels:", y.shape)

# Shuffle
inds = np.arange(len(X))
np.random.shuffle(inds)
X = X[inds]
y = y[inds]

# Reshape for CNN: (samples, window_size, 1)
X = np.expand_dims(X, axis=-1)

############################################
# 5. BUILD & TRAIN A 1D CNN
############################################

model = keras.Sequential([
    keras.layers.Conv1D(16, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, 1)),
    keras.layers.MaxPooling1D(2),
    keras.layers.Dropout(0.2),
    
    keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling1D(2),
    keras.layers.Dropout(0.2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')  # output: anomaly or not
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Simple train/val split
split = int(len(X)*0.8)
X_train, y_train = X[:split], y[:split]
X_val,   y_val   = X[split:], y[split:]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

############################################
# 6. EXPORT MODEL TO TFLITE
############################################

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Model exported to {TFLITE_MODEL_PATH}")

############################################
# 7. SLIDE OVER THE FULL TELEMETRY
############################################
# We'll slide a window of size WINDOW_SIZE, get anomaly probability,
# and mark windows > ANOMALY_THRESHOLD as anomalies.

def sliding_windows(signal, window_size, step=1):
    windows = []
    indices = []
    i = 0
    while (i + window_size) <= len(signal):
        windows.append(signal[i:i+window_size])
        indices.append(i)
        i += step
    return np.array(windows), np.array(indices)

test_windows, test_indices = sliding_windows(tele_vals, WINDOW_SIZE, step=1)
test_windows = np.expand_dims(test_windows, axis=-1)

y_scores = model.predict(test_windows).flatten()
y_pred   = (y_scores > ANOMALY_THRESHOLD).astype(int)
print("Inference windows:", len(y_pred))

############################################
# 8. PLOT RESULTS (Mark anomalies in red)
############################################

# Normal band
mean_val = np.mean(tele_vals)
std_val  = np.std(tele_vals)
lower_band = mean_val - NORMAL_BAND_MULT * std_val
upper_band = mean_val + NORMAL_BAND_MULT * std_val

plt.figure(figsize=(12,6))
plt.title("Telemetry with Detected Short-Pattern Anomalies (NN)")

# Entire telemetry in blue
plt.plot(tele_time, tele_vals, color='blue', label='Telemetry')

# Yellow band
plt.axhspan(lower_band, upper_band, color='yellow', alpha=0.3,
            label=f"Normal band ±{NORMAL_BAND_MULT}σ")

# Mark anomaly windows in red
for window_idx, is_anom in enumerate(y_pred):
    if is_anom == 1:
        start_idx = test_indices[window_idx]
        end_idx   = start_idx + WINDOW_SIZE - 1
        if end_idx >= len(tele_time):
            end_idx = len(tele_time) - 1
        
        plt.axvspan(tele_time[start_idx], tele_time[end_idx], color='red', alpha=0.3)

plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

print("\nAdjusting Sensitivity:")
print("- If no red areas appear, lower ANOMALY_THRESHOLD (currently 0.5).")
print("- If too many red areas, raise it above 0.5.")
print("- 23-sample pattern might not be strongly distinctive; consider tweaking EPOCHS, CNN layers, etc.")