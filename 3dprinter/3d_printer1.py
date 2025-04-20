# "data/000052 10_04_2025_norm.txt"
import numpy as np
import matplotlib.pyplot as plt

def read_acceleration_file(filename):
    """
    Reads acceleration data from a file.
    Expects each row to contain 4 values separated by semicolons:
      X; Y; Z; timestamp
    The timestamp is assumed to be in milliseconds.
    
    Returns:
      t: time array in seconds (relative to first timestamp)
      x, y, z: acceleration values for each axis.
    """
    try:
        # Load data from file using semicolon as the delimiter.
        data = np.loadtxt(filename, delimiter=";")
        # Extract columns: 0 = X, 1 = Y, 2 = Z, 3 = timestamp.
        t = data[:, 3]
        # Convert timestamp to seconds relative to the first measurement.
        t_rel = (t - t[0]) / 1000.0
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        return t_rel, x, y, z
    except Exception as e:
        print(f"Error reading '{filename}': {e}")
        return None, None, None, None

def main():
    # Filenames for normal and fail data.
    normal_file = "data/000052 10_04_2025_norm.txt"
    fail_file = "data/001223 10_04_2025_loose.txt"
    
    # normal_file = "data/133112 10_04_2025_norm_sosige.txt"
    # fail_file = "data/133301 10_04_2025loose_sosige.txt"
    
    # Attempt to load the normal file.
    t_normal, x_normal, y_normal, z_normal = read_acceleration_file(normal_file)
    # Attempt to load the fail file.
    t_fail, x_fail, y_fail, z_fail = read_acceleration_file(fail_file)
    
    # If any file cannot be loaded, use sample data.
    if t_normal is None:
        # Sample data for normal accelerations.
        sample_normal = """-256.18;11456.25;961;1744232452862
-275.33;11465.83;962;1744232452890
-260.00;11460.00;960;1744232452900
-270.00;11470.00;963;1744232452920
-265.00;11462.00;961;1744232452940"""
        from io import StringIO
        data = np.loadtxt(StringIO(sample_normal), delimiter=";")
        t = (data[:, 3] - data[0, 3]) / 1000.0
        x_normal = data[:, 0]
        y_normal = data[:, 1]
        z_normal = data[:, 2]
        t_normal = t

    if t_fail is None:
        # Sample data for fail accelerations.
        sample_fail = """-300.00;11450.00;950;1744232452862
-310.00;11455.00;955;1744232452890
-305.00;11453.00;952;1744232452900
-315.00;11458.00;957;1744232452920
-308.00;11454.00;951;1744232452940"""
        from io import StringIO
        data = np.loadtxt(StringIO(sample_fail), delimiter=";")
        t = (data[:, 3] - data[0, 3]) / 1000.0
        x_fail = data[:, 0]
        y_fail = data[:, 1]
        z_fail = data[:, 2]
        t_fail = t

    # Create one figure that will contain both normal and fail curves.
    plt.figure(figsize=(12, 6))

    # Plot Normal data with bright (solid) colors.
    plt.plot(t_normal, x_normal, color='red',   linewidth=2, label="Normal X")
    plt.plot(t_normal, y_normal, color='green', linewidth=2, label="Normal Y")
    plt.plot(t_normal, z_normal, color='blue',  linewidth=2, label="Normal Z")

    # Plot Fail data with the same colors but faded (using alpha and dashed lines).
    # plt.plot(t_fail, x_fail, color='red',   linewidth=2, alpha=0.5, linestyle="--", label="Fail X")
    # plt.plot(t_fail, y_fail, color='green', linewidth=2, alpha=0.5, linestyle="--", label="Fail Y")
    # plt.plot(t_fail, z_fail, color='blue',  linewidth=2, alpha=0.5, linestyle="--", label="Fail Z")

    # Add labels, title, legend, and grid.
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.title("XYZ Accelerations: Normal vs Fail")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
