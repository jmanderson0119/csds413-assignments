import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    data = pd.read_csv('../datasets/cars.csv', header=None)
    us_mileage = data[0].dropna().to_numpy()
    jp_mileage = data[1].dropna().to_numpy()

    sns.set_style("whitegrid")
    sns.set_palette("muted")
    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    percentiles = np.linspace(0, 100, 75)
    norm_quantiles_us = np.percentile(np.random.normal(loc=np.mean(us_mileage), 
    scale=np.std(us_mileage), size=10_000), percentiles)
    sample_quantiles_us = np.percentile(us_mileage, percentiles)

    norm_quantiles_jp = np.percentile(np.random.normal(loc=np.mean(jp_mileage), 
    scale=np.std(jp_mileage), size=10_000), percentiles)
    sample_quantiles_jp = np.percentile(jp_mileage, percentiles)

    axes[0].scatter(norm_quantiles_us, sample_quantiles_us, alpha=0.5, 
                    s=20, color='blue', edgecolor='blue')
    axes[0].plot([min(norm_quantiles_us), max(norm_quantiles_us)],
                 [min(norm_quantiles_us), max(norm_quantiles_us)],
                 'k--', lw=2, alpha=0.7)
    axes[0].set_title("US Car Mileage Quantiles", fontsize=14)
    axes[0].set_xlabel("Theoretical Quantiles", fontsize=12)
    axes[0].set_ylabel("Sample Quantiles", fontsize=12)

    axes[1].scatter(norm_quantiles_jp, sample_quantiles_jp, alpha=0.5, 
                    s=20, color='purple', edgecolor='purple')
    axes[1].plot([min(norm_quantiles_jp), max(norm_quantiles_jp)],
                 [min(norm_quantiles_jp), max(norm_quantiles_jp)],
                 'k--', lw=2, alpha=0.7)
    axes[1].set_title("Japanese Car Mileage Quantiles", fontsize=14)
    axes[1].set_xlabel("Theoretical Quantiles", fontsize=12)
    axes[1].set_ylabel("Sample Quantiles", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"../figures/qqplot.png", bbox_inches='tight', dpi=300)

if __name__ == "__main__": main()
