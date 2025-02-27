import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# MARK: 10
############################################
# 1. PARAMETERS
############################################

PATTERN_CSV = "data_phone/slam_two_peaks_pattern.csv"
TELEMETRY_CSV = "data_phone/one_slam.csv"

WINDOW_SIZE = 50    # Each training window length
POSITIVE_SAMPLES = 200
NEGATIVE_SAMPLES = 300

EPOCHS = 10
BATCH_SIZE = 32

# Get the current date and time
current_datetime = datetime.now()

# Format the date and time
formatted_datetime = current_datetime.strftime("%m%d_%H%M%S")

TFLITE_MODEL_PATH = f"tfmodels/model_{formatted_datetime}.tflite"

# For final detection
ANOMALY_THRESHOLD = 0.5  # Probability above this => pattern present
NORMAL_BAND_MULT = 2.0   # For yellow band ± 2*std

############################################
# 2. HELPER FUNCTIONS
############################################

def load_csv(filename):
    df = pd.read_csv(filename)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.dropna(subset=["time","value"], inplace=True)
    return df["time"].values, df["value"].values

def random_window(signal, window_size):
    N = len(signal)
    start_idx = np.random.randint(0, N - window_size)
    return np.copy(signal[start_idx : start_idx + window_size])

def embed_pattern_in_window(signal, pattern, window_size):
    """
    Picks a random window from 'signal' of length `window_size`
    and embeds the entire `pattern` at a random offset inside that window.
    """
    w = random_window(signal, window_size)
    p_len = len(pattern)
    if p_len <= window_size:
        insert_idx = np.random.randint(0, window_size - p_len + 1)
        w[insert_idx:insert_idx + p_len] += pattern
    else:
        # If pattern is bigger than the window, just truncate for this demo
        w = pattern[:window_size]
    return w

############################################
# 3. LOAD PATTERN & TELEMETRY
############################################
pattern_time, pattern_vals = load_csv(PATTERN_CSV)
tele_time, tele_vals       = load_csv(TELEMETRY_CSV)

pattern_len = len(pattern_vals)
tele_len    = len(tele_vals)

print(f"Pattern length: {pattern_len}")
print(f"Telemetry length: {tele_len}")

############################################
# 4. CREATE TRAINING DATASET
############################################
X_pos, y_pos = [], []
for _ in range(POSITIVE_SAMPLES):
    window = embed_pattern_in_window(tele_vals, pattern_vals, WINDOW_SIZE)
    X_pos.append(window)
    y_pos.append(1)

X_neg, y_neg = [], []
for _ in range(NEGATIVE_SAMPLES):
    window = random_window(tele_vals, WINDOW_SIZE)
    X_neg.append(window)
    y_neg.append(0)

X = np.array(X_pos + X_neg)
y = np.array(y_pos + y_neg)
print("Dataset shape:", X.shape, "Labels shape:", y.shape)

# Shuffle
inds = np.arange(len(X))
np.random.shuffle(inds)
X = X[inds]
y = y[inds]

# Reshape for CNN: (samples, window_size, 1)
X = np.expand_dims(X, axis=-1)

############################################
# 5. BUILD & TRAIN THE 1D CNN
############################################
model = keras.Sequential([
    keras.layers.Conv1D(16, 3, activation='relu', input_shape=(WINDOW_SIZE, 1)),
    keras.layers.MaxPooling1D(2),
    keras.layers.Dropout(0.2),

    keras.layers.Conv1D(32, 3, activation='relu'),
    keras.layers.MaxPooling1D(2),
    keras.layers.Dropout(0.2),

    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train / Validate split
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
# 7. SLIDING WINDOW INFERENCE
############################################
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

y_scores = model.predict(test_windows).flatten()  # probability of pattern
y_pred   = (y_scores > ANOMALY_THRESHOLD).astype(int)

############################################
# 8. PLOT + MARK ONLY THE PATTERN REGION
############################################
mean_val = np.mean(tele_vals)
std_val  = np.std(tele_vals)
lower_band = mean_val - NORMAL_BAND_MULT * std_val
upper_band = mean_val + NORMAL_BAND_MULT * std_val

plt.figure(figsize=(12,6))
plt.title("Telemetry + Highlighted Pattern Region (Neural Net)")

# Plot telemetry in blue
plt.plot(tele_time, tele_vals, color='blue', label="Telemetry")

# Highlight normal amplitude band in yellow
plt.axhspan(lower_band, upper_band, color='yellow', alpha=0.3, label=f"Normal ±{NORMAL_BAND_MULT}σ")

# We know the pattern length is 'pattern_len'.
# If a window is predicted anomaly, we highlight from `start_idx`
# to `start_idx + pattern_len`.
for w_idx, is_anom in enumerate(y_pred):
    if is_anom == 1:
        start_idx = test_indices[w_idx]
        end_idx   = start_idx + pattern_len - 1
        if end_idx >= len(tele_time):
            end_idx = len(tele_time) - 1
        
        # Mark the region in red
        plt.axvspan(tele_time[start_idx], tele_time[end_idx], color='red', alpha=0.3)

plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

print("Done. Windows predicted as anomaly are highlighted only up to 'pattern_len' in telemetry.")