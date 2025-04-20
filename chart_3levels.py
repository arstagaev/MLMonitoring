import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def main():
    np.random.seed(42)

    # -------------------------------
    # 1) Synthetic time-series setup with new scale:
    #    Mean = 2.0 mm/s, 1σ = 2.0 mm/s.
    # -------------------------------
    n_points = 300
    true_mean = 2.0
    true_std = 2.0

    # x axis from 1000 to 3000.
    x = np.linspace(1000, 3000, n_points)
    # Generate data from a normal distribution with the new parameters.
    data = np.random.normal(loc=true_mean, scale=true_std, size=n_points)
    # Add a small sinusoidal pattern using the index.
    data += 0.2 * np.sin(0.1 * np.arange(n_points))
    
    # Manually adjust some points to simulate peaks:
    #   - data[50] is set to -5.0, which is below (mean - 3σ) = -4.0  → red peak.
    #   - data[200] is set to 7.0, which is between (mean + 2σ)=6.0 and (mean + 3σ)=8.0 → orange peak.
    #   - data[100] is set to 9.0, which is above (mean + 3σ)=8.0  → red peak.
    data[50] = -5.0  
    data[200] = 7.0  
    data[100] = 9.0

    # Calculate sample mean & std (for display, though our thresholds use the true_std)
    mean_val = np.mean(data)
    # For consistent threshold markings, we use the predefined sigma (true_std).
    # Define thresholds based on our design:
    y_mean = mean_val
    y_1p   = mean_val + true_std    # mean + 1σ (≈ 2.0 + 2.0 = 4.0 ideally)
    y_1m   = mean_val - true_std    # mean - 1σ (≈ 2.0 - 2.0 = 0.0)
    y_2p   = mean_val + 2 * true_std  # mean + 2σ (≈ 6.0)
    y_2m   = mean_val - 2 * true_std  # mean - 2σ (≈ -2.0)
    y_3p   = mean_val + 3 * true_std  # mean + 3σ (≈ 8.0)
    y_3m   = mean_val - 3 * true_std  # mean - 3σ (≈ -4.0)

    # ---------------------------------------------------------------
    # 2) Create figure with two subplots: time-series (left), PDF (right)
    # ---------------------------------------------------------------
    fig = plt.figure(figsize=(9, 5), facecolor='w')
    gs  = GridSpec(nrows=1, ncols=2, width_ratios=[3, 1], wspace=0.05)
    
    # Left subplot: Time series
    ax_ts = fig.add_subplot(gs[0])
    # Right subplot: Distribution (sharing the y-axis for alignment)
    ax_dist = fig.add_subplot(gs[1], sharey=ax_ts)

    # --------------------------------
    # 3) Plot the time-series on the left
    # --------------------------------
    ax_ts.plot(x, data, color='steelblue', lw=1.2, label="Measurement")

    # Draw horizontal lines for the mean and sigma levels.
    # Swapped colors: 
    #   - Green for ±1σ,
    #   - Orange for ±2σ,
    #   - Red for ±3σ.
    ax_ts.axhline(y_mean, color='black', linestyle='-',  label="Mean")
    ax_ts.axhline(y_1p,   color='green', linestyle='--', label="±1σ")
    ax_ts.axhline(y_1m,   color='green', linestyle='--')
    ax_ts.axhline(y_2p,   color='orange', linestyle='--', label="±2σ")
    ax_ts.axhline(y_2m,   color='orange', linestyle='--')
    ax_ts.axhline(y_3p,   color='red', linestyle='--', label="±3σ")
    ax_ts.axhline(y_3m,   color='red', linestyle='--')

    # Mark peaks with separated colors:
    # Red peaks: data points beyond ±3σ.
    mask_red = (data >= y_3p) | (data <= y_3m)
    # Orange peaks: data points beyond ±2σ but not beyond ±3σ.
    mask_orange = (((data >= y_2p) | (data <= y_2m)) & ~mask_red)

    ax_ts.scatter(x[mask_orange], data[mask_orange], color='orange', s=50, marker='o', label="Пик (±2σ)")
    ax_ts.scatter(x[mask_red], data[mask_red], color='red', s=50, marker='o', label="Пик (±3σ)")

    ax_ts.set_xlabel("Время (миллисекунды)")
    ax_ts.set_ylabel("Виброскорость (мм/сек)")
    ax_ts.set_title("Замер вибрации с ±1σ, ±2σ, ±3σ")
    ax_ts.set_xlim(1000, 3000)
    ax_ts.legend(loc="lower left")

    # -------------------------------------------------------
    # 4) Plot the normal (Gaussian) distribution on the right
    # -------------------------------------------------------
    # Set up the y-values for the PDF plot: use a range a bit wider than ±3σ.
    y_min = y_3m - 0.5 * true_std
    y_max = y_3p + 0.5 * true_std
    y_vals = np.linspace(y_min, y_max, 300)

    # Calculate the Gaussian PDF:
    pdf = (1.0 / (true_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((y_vals - mean_val) / true_std)**2)

    # Plot the PDF horizontally.
    ax_dist.plot(pdf, y_vals, color='gray', lw=2)
    ax_dist.fill_betweenx(y_vals, 0, pdf, color='gray', alpha=0.2)
    
    # Draw the same horizontal sigma lines on the PDF plot.
    ax_dist.axhline(y_mean, color='black',  linestyle='-')
    ax_dist.axhline(y_1p,   color='green',  linestyle='--')
    ax_dist.axhline(y_1m,   color='green',  linestyle='--')
    ax_dist.axhline(y_2p,   color='orange', linestyle='--')
    ax_dist.axhline(y_2m,   color='orange', linestyle='--')
    ax_dist.axhline(y_3p,   color='red',    linestyle='--')
    ax_dist.axhline(y_3m,   color='red',    linestyle='--')

    # Remove x-axis tick labels on the distribution plot.
    ax_dist.set_xticklabels([])
    ax_dist.set_xlim(left=0)
    ax_dist.set_title("Нормальное распределение\n(Правило 3σ)")

    # Optionally, adjust the y-limits for a neat display.
    min_data = min(data.min(), y_3m) - 0.5 * true_std
    max_data = max(data.max(), y_3p) + 0.5 * true_std
    ax_ts.set_ylim(min_data, max_data)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
