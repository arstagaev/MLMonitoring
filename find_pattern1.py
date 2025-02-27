import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


############################################
# 1. PARAMETERS
############################################

PATTERN_CSV = "data_phone/slam_two_peaks_pattern.csv"
TELEMETRY_CSV = "data_phone/two_slams.csv"

WINDOW_SIZE = 50    # length of each training window
POSITIVE_SAMPLES = 200
NEGATIVE_SAMPLES = 300

EPOCHS = 10
BATCH_SIZE = 32

TFLITE_MODEL_PATH = "pattern_detector.tflite"

# For final detection
ANOMALY_THRESHOLD = 0.7  # If model output > 0.5 => anomaly
NORMAL_BAND_MULT = 2.0   # For plotting normal band (±2 std around mean)

############################################
# 2. HELPER FUNCTIONS
############################################

def load_csv(filename):
    """
    Reads a CSV with columns [time, value] and returns:
    time (np.array), values (np.array).
    Ensures numeric, drops NaN.
    """
    df = pd.read_csv(filename)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.dropna(subset=["time","value"], inplace=True)
    
    return df["time"].values, df["value"].values

def random_window(signal, window_size):
    """
    Select a random window of length 'window_size' from 'signal'.
    """
    N = len(signal)
    start_idx = np.random.randint(0, N - window_size)
    return np.copy(signal[start_idx : start_idx + window_size])

def embed_pattern_in_window(signal, pattern, window_size):
    """
    Select a random window of length 'window_size' from 'signal'
    and embed 'pattern' somewhere inside that window.
    """
    w = random_window(signal, window_size)
    p_len = len(pattern)
    if p_len <= window_size:
        insert_idx = np.random.randint(0, window_size - p_len + 1)
        w[insert_idx : insert_idx + p_len] += pattern
    else:
        # If pattern is bigger than the window, just truncate for demo
        w = pattern[:window_size]
    return w

############################################
# 3. LOAD THE PATTERN & TELEMETRY
############################################

pattern_time, pattern_vals = load_csv(PATTERN_CSV)
tele_time, tele_vals       = load_csv(TELEMETRY_CSV)

print(f"Pattern length: {len(pattern_vals)}")
print(f"Telemetry length: {len(tele_vals)}")

############################################
# 4. BUILD A TRAINING DATASET
############################################
# We'll create 'positive' windows that contain the pattern
# and 'negative' windows that do not.

X_pos = []
y_pos = []
for _ in range(POSITIVE_SAMPLES):
    w = embed_pattern_in_window(tele_vals, pattern_vals, WINDOW_SIZE)
    X_pos.append(w)
    y_pos.append(1)

X_neg = []
y_neg = []
for _ in range(NEGATIVE_SAMPLES):
    w = random_window(tele_vals, WINDOW_SIZE)
    X_neg.append(w)
    y_neg.append(0)

X = np.array(X_pos + X_neg)
y = np.array(y_pos + y_neg)
print("Dataset shape:", X.shape, "Labels shape:", y.shape)

# Shuffle
idx = np.arange(len(X))
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# Reshape for CNN: (samples, window_size, 1)
X = np.expand_dims(X, axis=-1)

############################################
# 5. BUILD & TRAIN A 1D CNN
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
# 6. EXPORT MODEL TO TFLITE (FUTURE ANDROID USAGE)
############################################
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Model exported to {TFLITE_MODEL_PATH}")

############################################
# 7. SLIDE OVER FULL TELEMETRY TO FIND PATTERN
############################################
# We apply the trained model to overlapping windows across
# the entire telemetry signal to see where pattern is detected.

def sliding_windows(signal, window_size, step=1):
    """
    Returns all overlapping windows of size `window_size`,
    plus the start indices.
    """
    windows = []
    indices = []
    i = 0
    while (i + window_size) <= len(signal):
        windows.append(signal[i : i + window_size])
        indices.append(i)
        i += step
    return np.array(windows), np.array(indices)

test_windows, test_indices = sliding_windows(tele_vals, WINDOW_SIZE, step=1)
test_windows = np.expand_dims(test_windows, axis=-1)  # shape (num_windows, window_size, 1)

y_scores = model.predict(test_windows).flatten()  # array of probabilities
y_pred   = (y_scores > ANOMALY_THRESHOLD).astype(int)

############################################
# 8. PLOT RESULTS (MARK DETECTED ANOMALIES)
############################################

mean_val = np.mean(tele_vals)
std_val  = np.std(tele_vals)
lower_band = mean_val - NORMAL_BAND_MULT * std_val
upper_band = mean_val + NORMAL_BAND_MULT * std_val

plt.figure(figsize=(12,6))
plt.title("Telemetry with Detected Pattern (Neural Network)")

# Plot telemetry
plt.plot(tele_time, tele_vals, color='blue', label="Telemetry")

# Yellow band for normal amplitude range
plt.axhspan(lower_band, upper_band, color='yellow', alpha=0.3,
            label=f"Normal band ±{NORMAL_BAND_MULT}σ")

# Mark anomalies in red
for w_idx, is_anom in enumerate(y_pred):
    if is_anom == 1:
        start_idx = test_indices[w_idx]
        end_idx = start_idx + WINDOW_SIZE - 1
        if end_idx >= len(tele_time):
            end_idx = len(tele_time) - 1
        
        plt.axvspan(tele_time[start_idx], tele_time[end_idx], color='red', alpha=0.3)

plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

print("Detection complete. Adjust ANOMALY_THRESHOLD for sensitivity.")