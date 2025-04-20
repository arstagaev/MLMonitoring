import math
import time
import numpy as np
import matplotlib.pyplot as plt

# ============================
# SINE WAVE PARAMETERS
# ============================
# Define a list of sine waves as tuples: (frequency in Hz, amplitude, phase in radians)
sine_params = [
    (16.0, 1.0, 0),             # Sine 1: 16 Hz, amplitude 1.0, phase 0
    (8.0, 0.5, math.pi / 4),     # Sine 2: 8 Hz, amplitude 0.5, phase π/4
    (4.0, 0.2, math.pi / 2)      # Sine 3: 4 Hz, amplitude 0.2, phase π/2
]

# Sampling parameters
sample_freq = 100     # Sampling frequency in Hz (100 samples per second)
dt = 1.0 / sample_freq  # Time interval between samples (0.01 sec)
num_lines = 500       # Total number of samples to generate

# ============================
# Part 1: Generate the data file with summed sine wave
# ============================
start_time = time.time()  # Base timestamp
filename = "data.txt"

with open(filename, "w") as file:
    for i in range(num_lines):
        t = i * dt  # Relative time for current sample

        # Compute each sine value and sum them
        sine_values = [amp * math.sin(2 * math.pi * freq * t + phase)
                       for freq, amp, phase in sine_params]
        summed_signal = sum(sine_values)

        # Optionally, you could write the individual sine values as well,
        # here we write the summed signal and the timestamp.
        timestamp = start_time + t
        file.write(f"{summed_signal};{timestamp}\n")

print(f"Data generated in file: {filename}")

# ============================
# Part 2: Plot the summed sine wave and its FFT
# ============================
timestamps = []
signal_data = []

with open(filename, "r") as file:
    for line in file:
        parts = line.strip().split(";")
        if len(parts) >= 2:
            try:
                val = float(parts[0])
                ts = float(parts[1])
                signal_data.append(val)
                timestamps.append(ts)
            except ValueError:
                continue

# Convert lists to numpy arrays.
signal_array = np.array(signal_data)
timestamps = np.array(timestamps)

# Use the known sampling interval if timestamps are consistent.
if len(timestamps) >= 2:
    dt = timestamps[1] - timestamps[0]
else:
    dt = 0.01

# Compute the FFT of the summed signal.
fft_result = np.fft.fft(signal_array)
freqs = np.fft.fftfreq(len(signal_array), d=dt)
magnitude = np.abs(fft_result)

# Keep only the positive frequencies.
half_n = len(freqs) // 2
positive_freqs = freqs[:half_n]
positive_magnitude = magnitude[:half_n]

# Plot both the time domain signal (summed sine) and its FFT using subplots.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Time domain plot
ax1.plot(timestamps - timestamps[0], signal_array, color='blue')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.set_title("Summed Sine Wave")
ax1.grid(True)

# Frequency domain plot (FFT)
ax2.plot(positive_freqs, positive_magnitude, color='red')
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
ax2.set_title("FFT of Summed Sine Wave")
ax2.grid(True)

plt.tight_layout()
plt.show()
