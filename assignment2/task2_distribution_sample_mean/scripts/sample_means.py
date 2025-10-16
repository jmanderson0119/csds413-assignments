import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_sample_means(replicates_csv: str) -> np.ndarray:
    """
    Computes the sample mean for each replicate.
    
    :param replicates_csv: Path to CSV containing replicates
    :type replicates_csv: str
    :returns: Array of sample means, one per replicate
    :rtype: np.ndarray
    """
    df = pd.read_csv(replicates_csv)
    sample_means = df.mean(axis=0).to_numpy()
    return sample_means

def save_sample_means(sample_means: np.ndarray, output_csv: str):
    """
    Saves sample means to CSV file.
    
    :param sample_means: Array of sample means
    :type sample_means: np.ndarray
    :param output_csv: Output filename
    :type output_csv: str
    """
    df = pd.DataFrame({'sample_mean': sample_means})
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

def plot_sample_means_distribution(sample_means: np.ndarray, dist_name: str, 
                                   replicate_size: int, output_filename: str):
    """
    Plots the distribution of sample means.
    
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
                     edgecolor='white', alpha=0.7)
    
    ax.set_xlabel('Sample Mean', fontsize=14, labelpad=15)
    ax.set_ylabel('Density', fontsize=14, labelpad=15)
    ax.set_title(f'Distribution of {dist_name} Sample Means of Replicate with size {replicate_size}',
                fontsize=16, pad=20)
    
    mean_of_means = np.mean(sample_means)
    ax.axvline(mean_of_means, color='red', linestyle='--', linewidth=2,
              label=f'Mean of means: {mean_of_means:.3f}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_filename}", bbox_inches='tight', dpi=300)

def main():
    distributions = ['uniform', 'powerlaw']
    dist_labels = ['Uniform(0.25, 1.25)', 'Pareto(a=2.3)']
    sample_sizes = [10, 100, 1000]
    
    for dist, label in zip(distributions, dist_labels):
        for n in sample_sizes:
            input_csv = f'../datasets/{dist}_replicates_n{n}.csv'
            sample_means = compute_sample_means(input_csv)
            output_csv = f'../datasets/{dist}_sample_means_n{n}.csv'
            save_sample_means(sample_means, output_csv)
            plot_sample_means_distribution(sample_means, label, n, f'{dist}_sample_means_n{n}.png')
    
if __name__ == "__main__": main()
