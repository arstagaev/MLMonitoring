import pandas as pd
import numpy as np
import keras
from keras import layers, models
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the vibration telemetry data
# Replace 'vibration_logs.csv' with your actual CSV file
df = pd.read_csv("plain1_anomaly.csv")

# Ensure time is the index and drop unnecessary columns if needed
df = df[['time', 'value']]
df.set_index('time', inplace=True)

# Normalize the vibration data
scaler = StandardScaler()
df['value_normalized'] = scaler.fit_transform(df[['value']])

# Parameters
TIME_STEPS = 50  # Number of time steps in each sequence

# Function to create sequences
def create_sequences(data, time_steps):
    sequences = []
    for i in range(len(data) - time_steps + 1):
        sequences.append(data[i:i+time_steps])
    return np.array(sequences)

# Prepare training data
train_data = df['value_normalized'].values
x_train = create_sequences(train_data, TIME_STEPS)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# Build the Autoencoder model
model = models.Sequential([
    layers.Input(shape=(TIME_STEPS, 1)),
    layers.Conv1D(32, 7, activation='relu', padding='same'),
    layers.Dropout(0.2),
    layers.Conv1D(16, 7, activation='relu', padding='same'),
    layers.Conv1DTranspose(16, 7, activation='relu', padding='same'),
    layers.Dropout(0.2),
    layers.Conv1DTranspose(32, 7, activation='relu', padding='same'),
    layers.Conv1DTranspose(1, 7, activation=None, padding='same')
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
history = model.fit(
    x_train,
    x_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Prepare test data
x_test = create_sequences(train_data, TIME_STEPS)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Get reconstruction loss for test data
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1).flatten()

# Determine anomaly threshold
threshold = np.percentile(test_mae_loss, 95)  # Adjust percentile as needed
print(f"Anomaly Threshold: {threshold}")

# Detect anomalies
anomalies = test_mae_loss > threshold

# Map anomalies back to original time index
anomaly_indices = np.where(anomalies)[0]
anomaly_timestamps = df.iloc[anomaly_indices + TIME_STEPS - 1].index

# Visualize anomalies
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'], label='Vibration Data')
plt.scatter(anomaly_timestamps, df.loc[anomaly_timestamps, 'value'], color='red', label='Anomalies')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Vibration Value")
plt.title("Vibration Telemetry with Anomalies")
plt.show()

# Save anomalies to CSV
anomaly_df = df.loc[anomaly_timestamps]
anomaly_df.to_csv("anomalies_detected.csv")
print("Anomalies saved to 'anomalies_detected.csv'")
