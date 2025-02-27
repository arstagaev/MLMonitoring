import csv
import matplotlib.pyplot as plt

def read_and_validate_csv(file_path):
    """
    Reads a CSV file, validates its format, and extracts the `value` column.

    :param file_path: Path to the CSV file.
    :return: A list of values from the `value` column.
    :raises ValueError: If the CSV file format is incorrect.
    """
    values = []

    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        if 'value' not in reader.fieldnames:
            raise ValueError("CSV file must contain a 'value' column.")

        for row in reader:
            try:
                values.append(float(row['value']))
            except ValueError:
                raise ValueError(f"Invalid value encountered: {row['value']}")

    return values

def plot_values(values):
    """
    Plots the `value` column from the CSV file.

    :param values: List of values to plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(values, label="Value", color="blue", marker="o")
    plt.title("CSV Values Chart", fontsize=16)
    plt.xlabel("Index", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace 'values.csv' with your CSV file path
    # file_path = 'data_phone/test_peak.csv'
    # file_path = 'data_phone/test.csv'
    # file_path = 'data_phone/slam_two_peaks.csv'
    # file_path = 'data_phone/two_slams.csv'
    # file_path = 'data_phone/raw_garbage.csv'
    file_path = 'data_syntetic/3peak_pattern.csv'
    try:
        values = read_and_validate_csv(file_path)
        print("CSV file format is valid. Plotting the chart...")
        plot_values(values)
    except ValueError as e:
        print(f"Error: {e}")
