# https://keras.io/examples/timeseries/timeseries_anomaly_detection/#timeseries-data-with-anomalies

import numpy as np
import pandas as pd
import keras
from keras import layers
from matplotlib import pyplot as plt


print("#############    NEW  ###########")
df_small_noise = pd.read_csv(
    "data/plain1.csv"
)

df_daily_jumpsup = pd.read_csv(
    "data/plain1_anomaly4_153.csv"
)

print(df_small_noise)

print(df_small_noise.head())
print(df_daily_jumpsup.head())


# fig, ax = plt.subplots()
# df_small_noise.plot(legend=False, ax=ax)
# plt.show()

# fig, ax = plt.subplots()
# df_daily_jumpsup.plot(legend=False, ax=ax)
# plt.show()
print(df_small_noise.columns)
print(df_daily_jumpsup.columns)
# Assuming your DataFrame is na
# Assuming your DataFrame is named df
# Create the plot
plt.figure(figsize=(10, 6))

# Plot data from the first DataFrame
plt.plot(df_small_noise['time'], df_small_noise['value'], label='Value 1', color='blue')

# Plot data from the second DataFrame
plt.plot(df_daily_jumpsup['time'], df_daily_jumpsup['value'], label='Value 2', color='red')


# df_small_noise.plot(x='time', y='value', kind='line', figsize=(10, 6), title="Time vs Value")
# df_daily_jumpsup.plot(x='time', y='value', kind='line', figsize=(10, 6), title="Time vs Value 2", color='r')
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid()
#plt.show()
#######################################################
TIME_STEPS = 100
## Prepare training data
# Normalize and save the mean and std we get,
# for normalizing test data.
training_mean = df_small_noise.mean()
training_std = df_small_noise.std()
df_training_value = (df_small_noise - training_mean) / training_std
print("Number of training samples:", len(df_training_value))

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
plt.show()

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
plt.show()
# JUST FOR FUN ENDS

################ Prepare test data
df_test_value = (df_daily_jumpsup - training_mean) / training_std
fig, ax = plt.subplots()
df_test_value.plot(legend=False, ax=ax)
# plt.show()



x_test = create_sequences(df_small_noise.values)


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
df_daily_jumpsup.plot(legend=False, ax=ax, color='b')
df_subset.plot(legend=False, ax=ax, color="r")

# Customize the plot
#plt.legend()
plt.title("Test Data with Anomalies")
plt.show()
print(anomalous_data_indices)

print("############  END  ############")