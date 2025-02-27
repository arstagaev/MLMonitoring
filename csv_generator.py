import numpy as np
import csv

def generate_sine_cosine(amplitude, period, length, sine_file, cosine_file):
    """
    Generate sine and cosine values with the given amplitude, period, and length,
    and save them to separate CSV files.

    :param amplitude: Amplitude of the sine and cosine functions.
    :param period: Period of the sine and cosine functions.
    :param length: Total number of points to generate.
    :param sine_file: Path to the output CSV file for sine values.
    :param cosine_file: Path to the output CSV file for cosine values.
    """
    # Generate time values
    time = np.arange(length)

    # Calculate angular frequency (omega)
    omega = 2 * np.pi / period

    # Generate sine and cosine values
    sine_values = amplitude * np.sin(omega * time)
    cosine_values = amplitude * np.cos(omega * time)

    # Write sine values to a CSV file
    with open(sine_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["time", "value"])
        # Write data rows
        for t, sine in zip(time, sine_values):
            writer.writerow([t, sine])

    # Write cosine values to a separate CSV file
    with open(cosine_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["time", "value"])
        # Write data rows
        for t, cosine in zip(time, cosine_values):
            writer.writerow([t, cosine])

    print(f"Sine values saved to {sine_file}")
    print(f"Cosine values saved to {cosine_file}")

# Example usage
sine_csv = "sine_output.csv"
cosine_csv = "cosine_output.csv"
generate_sine_cosine(amplitude=100.0, period=230, length=3400, sine_file=sine_csv, cosine_file=cosine_csv)
