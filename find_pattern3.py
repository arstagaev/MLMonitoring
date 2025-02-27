import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from datetime import datetime

print("############################################ NEW CALCULATIONS ############################################")
############################################
# 1. PARAMETERS
############################################

# The known two-peak anomaly pattern
PATTERN_CSV = "data_syntetic/3peak_pattern.csv"

# A list of negative CSV files (1 peak, 3 peaks, or random non-anomalous data)
NEGATIVE_CSV_LIST = [
    "data_phone/one_slam.csv",
    "data_phone/negative1.csv",
    "data_phone/negative2_one_slam.csv",
    "data_phone/negative3_triple_slam.csv",
    "data_syntetic/2peak.csv",
    "data_syntetic/1peak.csv",
    # Add more as needed
]

# Final telemetry to evaluate
# TELEMETRY_CSV = "data_phone/two_slams.csv"
# TELEMETRY_CSV = "data_phone/one_slam.csv"
# TELEMETRY_CSV = "data_syntetic/3peak.csv"
TELEMETRY_CSV = "data_syntetic/1peak.csv"

# Window size for training/inference
WINDOW_SIZE = 550  

# How many positive windows to create (by embedding the pattern)
POSITIVE_SAMPLES = 200

# Additional random negative windows from the final telemetry
RANDOM_NEG_SAMPLES = 300

# Model training hyperparams
EPOCHS = 10
BATCH_SIZE = 32

# Get the current date and time
current_datetime = datetime.now()

# Format the date and time
formatted_datetime = current_datetime.strftime("%m%d_%H%M%S")

TFLITE_MODEL_PATH = f"tfmodels/model_{formatted_datetime}.tflite"

# Detection threshold
ANOMALY_THRESHOLD = 0.0012 # Probability > 0.5 => anomaly
NORMAL_BAND_MULT = 2.0   # For plotting normal amplitude band

############################################
# 2. HELPER FUNCTIONS
############################################

def load_csv(filename):
    """
    Reads a CSV with columns [time, value] and returns:
    - time: numpy array of time indices
    - values: numpy array of float values
    """
    df = pd.read_csv(filename)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.dropna(subset=["time","value"], inplace=True)
    return df["time"].values, df["value"].values

def random_window(signal, window_size):
    """
    Selects a random window of length 'window_size' from 'signal'.
    """
    N = len(signal)
    start_idx = np.random.randint(0, N - window_size)
    return np.copy(signal[start_idx : start_idx + window_size])

def embed_pattern_in_window(signal, pattern, window_size):
    """
    Select a random window from 'signal' of length 'window_size'
    and embed the entire 'pattern' at a random offset in that window.
    """
    w = random_window(signal, window_size)
    p_len = len(pattern)
    if p_len <= window_size:
        insert_idx = np.random.randint(0, window_size - p_len + 1)
        w[insert_idx : insert_idx + p_len] += pattern
    else:
        # If pattern is longer than the window, just truncate (demo).
        w = pattern[:window_size]
    return w

def window_data(values, window_size):
    """
    Slide a window of size 'window_size' across 'values' (step=1).
    Returns a list of windows.
    """
    windows = []
    N = len(values)
    if N < window_size:
        # Optionally skip or pad:
        return []
    for start_idx in range(N - window_size + 1):
        w = values[start_idx : start_idx + window_size]
        windows.append(w)
    return windows

############################################
# 3. LOAD THE TWO-PEAK PATTERN
############################################
pattern_time, pattern_values = load_csv(PATTERN_CSV)
pattern_length = len(pattern_values)
print(f"Loaded two-peak pattern of length={pattern_length}")

############################################
# 4. BUILD NEGATIVE DATASET
############################################

# A) From multiple negative CSV files (each has 1-peak, 3-peaks, or random shapes)
X_neg, y_neg = [], []
for neg_csv in NEGATIVE_CSV_LIST:
    neg_time, neg_vals = load_csv(neg_csv)
    # We'll break them into sliding windows of length 'WINDOW_SIZE'.
    # Each is labeled negative (0).
    neg_windows = window_data(neg_vals, WINDOW_SIZE)
    for w in neg_windows:
        X_neg.append(w)
        y_neg.append(0)
print(f"Total negative windows from CSV files: {len(X_neg)}")

# B) Also create random negative windows from final telemetry
tele_time, tele_vals = load_csv(TELEMETRY_CSV)
for _ in range(RANDOM_NEG_SAMPLES):
    w = random_window(tele_vals, WINDOW_SIZE)
    X_neg.append(w)
    y_neg.append(0)
print(f"After adding random neg windows: {len(X_neg)}")

############################################
# 5. BUILD POSITIVE DATASET (EMBED PATTERN)
############################################
# We'll also use 'tele_vals' as the base to embed the known pattern
X_pos, y_pos = [], []
for _ in range(POSITIVE_SAMPLES):
    w = embed_pattern_in_window(tele_vals, pattern_values, WINDOW_SIZE)
    X_pos.append(w)
    y_pos.append(1)
print(f"Positive windows: {len(X_pos)}")

############################################
# 6. COMBINE & SHUFFLE
############################################
X = np.array(X_pos + X_neg)  # shape (num_samples, window_size)
y = np.array(y_pos + y_neg)  # shape (num_samples,)

# Shuffle
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Reshape for CNN: (samples, window_size, 1)
X = np.expand_dims(X, axis=-1)
print("Final dataset shape:", X.shape, "Labels shape:", y.shape)

############################################
# 7. BUILD & TRAIN A 1D CNN
############################################
model = keras.Sequential([
    keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, 1)),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Dropout(0.2),

    keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Dropout(0.2),

    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')  # 0 or 1
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train/validation split
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_val,   y_val   = X[split:], y[split:]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

############################################
# 8. EXPORT MODEL TO TFLITE
############################################
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)
print(f"TFLite model saved to {TFLITE_MODEL_PATH}")

############################################
# 9. SLIDE OVER THE FINAL TELEMETRY FOR DETECTION
############################################
# We'll re-use tele_time, tele_vals from above.

def sliding_windows(signal, window_size, step=1):
    windows = []
    indices = []
    i = 0
    while (i + window_size) <= len(signal):
        windows.append(signal[i : i + window_size])
        indices.append(i)
        i += step
    return np.array(windows), np.array(indices)

test_windows, test_indices = sliding_windows(tele_vals, WINDOW_SIZE, step=1)
test_windows = np.expand_dims(test_windows, axis=-1)

print("Running inference on final telemetry...")
y_scores = model.predict(test_windows).flatten()
y_pred   = (y_scores > ANOMALY_THRESHOLD).astype(int)
print(">>>>>>> score:", str(y_scores),"vs threshold:",ANOMALY_THRESHOLD)
# Filter the scores
filtered_scores = [score for score in y_scores if score > ANOMALY_THRESHOLD]

# Print the filtered scores
print(">>>>>>>>Scores greater than threshold:")
print(filtered_scores)

############################################
# 10. PLOT RESULTS
############################################
mean_val = np.mean(tele_vals)
std_val  = np.std(tele_vals)
lower_band = mean_val - NORMAL_BAND_MULT * std_val
upper_band = mean_val + NORMAL_BAND_MULT * std_val

plt.figure(figsize=(12,6))
plt.title("Telemetry with Two-Peak Anomaly Detection (NN)")

# Plot entire telemetry in blue
plt.plot(tele_time, tele_vals, color='blue', label="Telemetry")

# Mark normal band in yellow
plt.axhspan(lower_band, upper_band, color='yellow', alpha=0.3,
            label=f"Normal band ±{NORMAL_BAND_MULT}σ")

# Mark anomalies in red
# We'll highlight from start_idx to start_idx + pattern_length
# (assuming the pattern is that long inside the window).
for w_idx, is_anomaly in enumerate(y_pred):
    if is_anomaly == 1:
        start_idx = test_indices[w_idx]
        end_idx   = start_idx + pattern_length - 1
        if end_idx >= len(tele_time):
            end_idx = len(tele_time) - 1
        
        plt.axvspan(tele_time[start_idx], tele_time[end_idx], color='red', alpha=0.3)

plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
plt.close()
print("Done! Adjust ANOMALY_THRESHOLD or add more negative examples if needed.")