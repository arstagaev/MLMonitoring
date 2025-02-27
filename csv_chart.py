import csv
import matplotlib.pyplot as plt

# Function to read CSV file and extract data
def read_csv(file_path):
    time = []
    value = []

    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            time.append(int(row['time']))
            value.append(float(row['value']))

    return time, value

# Function to plot the chart
def plot_chart(time, value):
    plt.figure(figsize=(10, 6))
    plt.plot(time, value, label="Value", color="blue", marker="o")
    plt.title("Time vs Value", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Replace 'data.csv' with your CSV file path
    file_path = 'data_phone/two_slams.csv'
    time, value = read_csv(file_path)
    plot_chart(time, value)