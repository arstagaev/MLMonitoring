import math
import time
import numpy as np
import matplotlib.pyplot as plt

# ============================
# SINE WAVE PARAMETERS
# ============================
sine_freq = 16.0       # Main sine frequency in Hz (e.g., 1.0 Hz)
amplitude = 1.0       # Amplitude of the sine wave
phase1 = 0            # Phase offset for X channel (radians)
phase2 = 2 * math.pi / 3  # Phase offset for Y channel (radians)
phase3 = 4 * math.pi / 3  # Phase offset for Z channel (radians)

# Sampling parameters
sample_freq = 100     # Sampling frequency in Hz (100 samples per second)
dt = 1.0 / sample_freq  # Time interval between samples (0.01 sec)
num_lines = 500       # Total number of samples to generate

# ============================
# Part 1: Generate the data file
# ============================
start_time = time.time()  # Base timestamp
filename = "data.txt"

with open(filename, "w") as file:
    for i in range(num_lines):
        t = i * dt  # Relative time for current sample
        
        # Compute sine values using the managed parameters.
        x = amplitude * math.sin(2 * math.pi * sine_freq * t + phase1)
        y = amplitude * math.sin(2 * math.pi * sine_freq * t + phase2)
        z = amplitude * math.sin(2 * math.pi * sine_freq * t + phase3)
        
        timestamp = start_time + t
        file.write(f"{x};{y};{z};{timestamp}\n")

print(f"Data generated in file: {filename}")

# ============================
# Part 2: Plot the sine wave and its FFT
# ============================
# Read the data file to extract time stamps and X channel data.
timestamps = []
x_data = []

with open(filename, "r") as file:
    for line in file:
        parts = line.strip().split(";")
        if len(parts) >= 4:
            try:
                x_val = float(parts[0])
                ts = float(parts[3])
                x_data.append(x_val)
                timestamps.append(ts)
            except ValueError:
                continue

# Convert lists to numpy arrays.
x_array = np.array(x_data)
timestamps = np.array(timestamps)

# Use the known sampling interval if timestamps are consistent.
if len(timestamps) >= 2:
    dt = timestamps[1] - timestamps[0]
else:
    dt = 0.01

# Compute the FFT of the X channel.
fft_result = np.fft.fft(x_array)
freqs = np.fft.fftfreq(len(x_array), d=dt)
magnitude = np.abs(fft_result)

# Keep only the positive frequencies.
half_n = len(freqs) // 2
positive_freqs = freqs[:half_n]
positive_magnitude = magnitude[:half_n]

# Plot both the time domain sine wave and its FFT using subplots.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Time domain plot (sine wave)
ax1.plot(timestamps - timestamps[0], x_array, color='blue')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.set_title("Sine Wave (X Channel)")
ax1.grid(True)

# Frequency domain plot (FFT)
ax2.plot(positive_freqs, positive_magnitude, color='red')
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
ax2.set_title("FFT of Sine Wave")
ax2.grid(True)

plt.tight_layout()
plt.show()
