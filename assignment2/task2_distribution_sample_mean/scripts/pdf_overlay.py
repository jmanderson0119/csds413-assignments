import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

def overlay_normal(sample_means: np.ndarray, dist_name: str, 
                       replicate_size: int, output_filename: str):
    """
    Plots the histogram of sample means with overlaid normal PDF.
    
    :param sample_means: Array of sample means
    :type sample_means: np.ndarray
    :param dist_name: Name of the distribution
    :type dist_name: str
    :param replicate_size: Replicate size
    :type replicate_size: int
    :param output_filename: Output filename for figure
    :type output_filename: str
    """
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(10, 6))

    ax = sns.histplot(sample_means, bins=50, stat='density',
                      edgecolor='white', alpha=0.7, label='Sample Mean Distribution')

    mu = np.mean(sample_means)
    sigma = np.std(sample_means)
    x = np.linspace(min(sample_means), max(sample_means), 500)
    pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
    
    plt.plot(x, pdf, 'r--', label=f'Normal Approx. PDF\nmu={mu:.3f}, sigma={sigma:.3f}')
    
    ax.set_xlabel('Sample Mean', fontsize=14, labelpad=15)
    ax.set_ylabel('Density', fontsize=14, labelpad=15)
    ax.set_title(f'Normal Overlay for {dist_name} Sample Means of Replicate Size {replicate_size})',
                 fontsize=16, pad=20)
    
    plt.legend(fontsize=11)
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_filename}", bbox_inches='tight', dpi=300)

def main():
    distributions = ['uniform', 'powerlaw']
    dist_labels = ['Uniform(0.25, 1.25)', 'Pareto(a=2.3)']
    sample_sizes = [10, 100, 1000]

    for dist, label in zip(distributions, dist_labels):
        for n in sample_sizes:
            means_path = f'../datasets/{dist}_sample_means_n{n}.csv'
            sample_means = pd.read_csv(means_path)['sample_mean'].to_numpy()
            overlay_normal(sample_means, label, n, f'{dist}_sample_means_overlay_n{n}.png')

if __name__ == "__main__": main()
