import pandas as pd
import numpy as np
import keras
from keras import layers, models
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# Load reference logs (normal vibration data)
reference_logs = pd.read_csv("data/plain1.csv")
new_logs = pd.read_csv("data/plain1_anomaly5_3.impulses.csv")

# Normalize the data
def normalize_data(df):
    mean = df['value'].mean()
    std = df['value'].std()
    df['normalized_value'] = (df['value'] - mean) / std
    return df, mean, std

reference_logs, ref_mean, ref_std = normalize_data(reference_logs)
new_logs['normalized_value'] = (new_logs['value'] - ref_mean) / ref_std

# Apply a high-pass filter
def high_pass_filter(data, cutoff=10, fs=1000, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

new_logs['normalized_value'] = high_pass_filter(new_logs['normalized_value'].values)
# Generate sequences for training, also may called like RESOLUTION of search the anomaly
TIME_STEPS = 5

def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

x_train = create_sequences(reference_logs['normalized_value'].values.reshape(-1, 1))
x_test = create_sequences(new_logs['normalized_value'].values.reshape(-1, 1))

print(f"Training input shape: {x_train.shape}")
print(f"Test input shape: {x_test.shape}")

# Build the autoencoder model
# model = models.Sequential([
#     layers.Input(shape=(TIME_STEPS, 1)),
#     layers.Conv1D(32, 7, padding="same", strides=2, activation="relu"),
#     layers.Dropout(0.2),
#     layers.Conv1D(16, 7, padding="same", strides=2, activation="relu"),
#     layers.Conv1DTranspose(16, 7, padding="same", strides=2, activation="relu"),
#     layers.Dropout(0.2),
#     layers.Conv1DTranspose(32, 7, padding="same", strides=2, activation="relu"),
#     layers.Conv1DTranspose(1, 7, padding="same")
# ])
# model.compile(optimizer='adam', loss='mse')

# model = models.Sequential([
#     layers.Input(shape=(1, 1)),  # Input shape for TIME_STEPS = 1
#     layers.Conv1D(32, 1, padding="same", strides=1, activation="relu"),  # Use kernel_size=1 for single time step
#     layers.Dropout(0.2),
#     layers.Conv1D(16, 1, padding="same", strides=1, activation="relu"),
#     layers.Conv1DTranspose(16, 1, padding="same", strides=1, activation="relu"),
#     layers.Dropout(0.2),
#     layers.Conv1DTranspose(32, 1, padding="same", strides=1, activation="relu"),
#     layers.Conv1DTranspose(1, 1, padding="same")  # Final layer to reconstruct the input shape
# ])
model = models.Sequential([
    layers.Input(shape=(TIME_STEPS, 1)),
    layers.Conv1D(64, kernel_size=3, padding="same", strides=1, activation="relu"),
    layers.Conv1D(32, kernel_size=3, padding="same", strides=1, activation="relu"),
    layers.Dropout(0.2),
    layers.Conv1D(16, kernel_size=3, padding="same", strides=1, activation="relu"),
    layers.Conv1DTranspose(16, kernel_size=3, padding="same", strides=1, activation="relu"),
    layers.Conv1DTranspose(32, kernel_size=3, padding="same", strides=1, activation="relu"),
    layers.Conv1DTranspose(64, kernel_size=3, padding="same", strides=1, activation="relu"),
    layers.Conv1DTranspose(1, kernel_size=3, padding="same")
])

model.compile(optimizer='adam', loss='mse')

model.summary()

# Train the model
history = model.fit(
    x_train, x_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    #callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5)]
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
)

# Plot training and validation loss
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.legend()
plt.show()

# Get reconstruction loss threshold from reference logs
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
threshold = np.max(train_mae_loss)
#threshold = np.percentile(train_mae_loss, 99.7)  # Use the 95th percentile of training MAE loss

print(f"Reconstruction error threshold: {threshold} and train_mae_loss {train_mae_loss}")

# Detect anomalies in new logs
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

# Detect anomalies
anomalies = test_mae_loss > threshold
print(f"Number of anomaly samples: {np.sum(anomalies)}")

# Mark anomalies in the original test data
anomalous_indices = []
for i in range(TIME_STEPS - 1, len(new_logs) - TIME_STEPS + 1):
    if np.all(anomalies[i - TIME_STEPS + 1 : i]):
        anomalous_indices.append(i)

new_logs['anomaly'] = 0
new_logs.loc[anomalous_indices, 'anomaly'] = 1

# Visualize anomalies
plt.figure(figsize=(15, 5))

# Plot the new logs
plt.plot(new_logs['time'], new_logs['value'], label="New Vibration Data")

# Plot the reference data with magenta color
plt.plot(reference_logs['time'], reference_logs['value'], color='magenta', label="Reference Data")

plt.scatter(new_logs[new_logs['anomaly'] == 1]['time'], 
            new_logs[new_logs['anomaly'] == 1]['value'], 
            color='red', label="Anomalies")
plt.legend()
plt.show()

# Save results
new_logs.to_csv("anomalies_detected.csv", index=False)