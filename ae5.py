# https://keras.io/examples/timeseries/timeseries_anomaly_detection/#timeseries-data-with-anomalies

import numpy as np
import pandas as pd
import keras
from keras import layers
from matplotlib import pyplot as plt

print("#############    NEW  ###########")
master_url_root = "https://raw.githubusercontent.com/numenta/NAB/master/data/"

# with small noise
# df_small_noise_url_suffix = "artificialNoAnomaly/art_daily_small_noise.csv"
# df_small_noise_url = master_url_root + df_small_noise_url_suffix
# df_small_noise = pd.read_csv(
#     df_small_noise_url, parse_dates=True, index_col="timestamp"
# )
df_small_noise = pd.read_csv(
    "data/art_daily_small_noise.csv", parse_dates=True, index_col="timestamp"
)

# with ANOMALY
# df_daily_jumpsup_url_suffix = "artificialWithAnomaly/art_daily_jumpsup.csv"
# df_daily_jumpsup_url = master_url_root + df_daily_jumpsup_url_suffix
# df_daily_jumpsup = pd.read_csv(
#     df_daily_jumpsup_url, parse_dates=True, index_col="timestamp"
# )
df_daily_jumpsup = pd.read_csv(
    "data/art_daily_jumpsup_with_anomaly.csv", parse_dates=True, index_col="timestamp"
)

print(df_small_noise.head())

print(df_daily_jumpsup.head())




fig, ax = plt.subplots()
df_small_noise.plot(legend=False, ax=ax)
#plt.show()

fig, ax = plt.subplots()
df_daily_jumpsup.plot(legend=False, ax=ax)
#plt.show()



## Prepare training data
# Normalize and save the mean and std we get,
# for normalizing test data.
training_mean = df_small_noise.mean()
training_std = df_small_noise.std()
df_training_value = (df_small_noise - training_mean) / training_std
print("Number of training samples:", len(df_training_value))


TIME_STEPS = 288


# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


x_train = create_sequences(df_training_value.values)
print("Training input shape: ", x_train.shape)
print(x_train)

model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Conv1DTranspose(
            filters=16,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

# Train the model
# Please note that we are using x_train as both the input and the target since this is a reconstruction model.

history = model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)

# Let's plot training and validation loss to see how the training went.
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
# plt.show()

# Detecting anomalies
# Get train MAE loss.
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
# plt.show()

# Get reconstruction loss threshold.
threshold = np.max(train_mae_loss)
print("Reconstruction error threshold: ", threshold)

# JUST FOR FUN:
# Checking how the first sequence is learnt
plt.plot(x_train[0],'b')
plt.plot(x_train_pred[0],'r')
# plt.show()
# JUST FOR FUN ENDS


################ Prepare test data
df_test_value = (df_daily_jumpsup - training_mean) / training_std
fig, ax = plt.subplots()
df_test_value.plot(legend=False, ax=ax)
# plt.show()








def generate_sine_wave_df(num_rows=4000, start_time="2014-04-01 00:00:00", freq="5T", min_val=-1.0, max_val=0.0):
    """
    Generate a DataFrame containing a sine wave scaled between min_val and max_val.

    Parameters:
    - num_rows: int, the number of rows (timestamps) in the DataFrame.
    - start_time: str, the starting timestamp for the sine wave data.
    - freq: str, the frequency of timestamps (e.g., "5T" for 5-minute intervals).
    - min_val: float, the minimum value of the sine wave.
    - max_val: float, the maximum value of the sine wave.

    Returns:
    - pd.DataFrame with columns ['timestamp', 'value'].
    """
    # Generate timestamps
    timestamps = pd.date_range(start=start_time, periods=num_rows, freq=freq)

    # Generate sine wave values
    x = np.linspace(0, 2 * np.pi * num_rows / 1000, num_rows)  # Adjust the frequency of the sine wave
    sine_values = (max_val - min_val) / 2 * np.sin(x) + (max_val + min_val) / 2  # Scale to [min_val, max_val]

    # Create the DataFrame
    df = pd.DataFrame({"timestamp": timestamps, "value": sine_values})

    return df

# Example usage
df = generate_sine_wave_df()
print(df.head())

# Generate sine wave data
df_sine_wave = generate_sine_wave_df()

# Use the same training_mean and training_std from your model training
# Replace with actual values or compute from your training data
training_mean = 0.0  # Update if you saved the actual mean
training_std = 1.0   # Update if you saved the actual std

# Normalize the sine wave data
df_sine_wave_normalized = (df_sine_wave["value"] - training_mean) / training_std

# Create sequences for the sine wave data
TIME_STEPS = 288

def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)



x_test = create_sequences(df_sine_wave_normalized.values)

# Add a third dimension to match the training data shape
x_test = np.expand_dims(x_test, axis=2)





######### Create sequences from test values.
#print("XXXX",df_test_value.head())
#x_test = create_sequences(df_test_value.values)
print("Test input shape: ", x_test.shape)

#################
# Get test MAE loss.
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

# Detect all the samples which are anomalies.
anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))



# data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
anomalous_data_indices = []
for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)

#Let's overlay the anomalies on the original test data plot.

df_subset = df_daily_jumpsup.iloc[anomalous_data_indices]
fig, ax = plt.subplots()
df_daily_jumpsup.plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color="r")
# Plot the sine wave
df_sine_wave_normalized.plot(
    x="timestamp", 
    y="value", 
    ax=ax, 
    color="magenta", 
    label="Sine Wave (Normalized)"
)
# Customize the plot
plt.legend()
plt.title("Test Data with Anomalies and Sine Wave")
plt.show()


print("############  END  ############")