import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Generate synthetic data
np.random.seed(42)
normal_data = np.sin(np.linspace(0, 100, 500))  # Sine wave as normal pattern
anomaly_data = np.random.normal(0, 1, 100)  # Random noise as anomalies
data = np.concatenate([normal_data, anomaly_data])

print("####### normal_data: ",normal_data.size)
print(normal_data)
print("####### anomaly_data: ",anomaly_data.size)
print(anomaly_data)
print("#######")

# Prepare time-series data
sequence_length = 20
generator = TimeseriesGenerator(data, data, length=sequence_length, batch_size=32)

# Define the model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(generator, epochs=10)

# Save the model in TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)


# Enable SELECT_TF_OPS to support TensorFlow operations not natively supported by TFLite
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS     # Enable TF ops
]

# Enable resource variables to handle TensorList operations
converter.experimental_enable_resource_variables = True


tflite_model = converter.convert()

# Save the model to a file
with open("anomaly_detector.tflite", "wb") as f:
    f.write(tflite_model)
