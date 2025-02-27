import csv
import os

# Function to process a single CSV file
def process_csv(input_file, output_dir):
    # Get the output file name (same as input but in the output directory)
    output_file = os.path.join(output_dir, os.path.basename(input_file))

    # Read the input file
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Skip the header if present
        values = []

        for row in reader:
            try:
                # Attempt to convert the value to float
                value = float(row[0])
                values.append(value)
            except (ValueError, IndexError):
                # Skip rows with non-float values or missing data
                continue

    # Create the new content with time and value columns
    new_content = [["time", "value"]]  # Add header row
    for i, value in enumerate(values, start=1):
        new_content.append([i, value])

    # Write the new content to the output file
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(new_content)

    print(f"Processed file saved to {output_file}")

# Main function to process an array of CSV files
def process_csv_files(input_files, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each file in the input list
    for input_file in input_files:
        process_csv(input_file, output_dir)

# Example usage
if __name__ == "__main__":
    input_files = [
        # "data_phone/negative1.csv", 
        # "data_phone/negative2_one_slam.csv",
        # "data_phone/negative3_triple_slam.csv",
        "data_syntetic/1peak.csv",
        "data_syntetic/2peak.csv",
        # "data_syntetic/3peak.csv"
        # "data_syntetic/3peak_pattern.csv"
    ]  # List of input CSV files
    output_directory = "data_syntetic"  # Directory to save the output files

    process_csv_files(input_files, output_directory)