import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def ks_test_theoretical_normal(sample_means: np.ndarray, theoretical_mu: float,
                                          theoretical_se: float) -> tuple:
    """
    Performs KS test comparing sample means to their theoretical normal distribution.
    
    :param sample_means: Array of sample means
    :type sample_means: np.ndarray
    :param theoretical_mu: Theoretical mean from CLT
    :type theoretical_mu: float
    :param theoretical_se: Theoretical standard error from CLT
    :type theoretical_se: float
    :returns: Tuple of KS statistic, p-value
    :rtype: tuple
    """
    ks_stat, p_value = stats.kstest(sample_means, 
                                    lambda x: stats.norm.cdf(x, 
                                                             loc=theoretical_mu, 
                                                             scale=theoretical_se))
    
    return ks_stat, p_value

def plot_ks_comparison(replicate_sizes: list, uniform_stats: list, 
                       pareto_stats: list, output_filename: str):
    """
    Plots KS statistics for Uniform vs Pareto across replicate sizes.
    
    :param replicate_sizes: List of replicate sizes
    :type replicate_sizes: list
    :param uniform_stats: List of KS statistics for uniform distribution
    :type uniform_stats: list
    :param pareto_stats: List of KS statistics for pareto distribution
    :type pareto_stats: list
    :param output_filename: Output filename for figure
    :type output_filename: str
    """
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(10, 6))
    
    plt.plot(replicate_sizes, uniform_stats, marker='o', linewidth=1, 
             markersize=4, label='Uniform', color='blue')
    plt.plot(replicate_sizes, pareto_stats, marker='o', linewidth=1, 
             markersize=4, label='Pareto', color='purple')
    
    plt.xscale('log')
    plt.xlabel('Number of Replicates', fontsize=14, labelpad=15)
    plt.ylabel('KS Statistic', fontsize=14, labelpad=15)
    plt.title('KS Test: Comparing Convergence to Normal', 
              fontsize=16, pad=20)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_filename}", bbox_inches='tight', dpi=300)
    plt.close()

def main():
    
    replicate_sizes = [10, 100, 1000]
    
    uniform_results = []
    pareto_results = []
    uniform_ks_stats = []
    pareto_ks_stats = []
    
    for n in replicate_sizes:
        sample_means_csv = f'../datasets/uniform_sample_means_n{n}.csv'
        sample_means = pd.read_csv(sample_means_csv)['sample_mean'].to_numpy()
        
        ks_stat, p_value = ks_test_theoretical_normal(
            sample_means, 0.75, 0.029)
        
        uniform_ks_stats.append(ks_stat)
        uniform_results.append({
            'distribution': 'Uniform',
            'replicate_size': n,
            'ks_statistic': ks_stat,
            'p_value': p_value
        })
    
    for n in replicate_sizes:
        sample_means_csv = f'../datasets/powerlaw_sample_means_n{n}.csv'
        sample_means = pd.read_csv(sample_means_csv)['sample_mean'].to_numpy()
        
        ks_stat, p_value = ks_test_theoretical_normal(
            sample_means, 0.769, 0.140)
        
        pareto_ks_stats.append(ks_stat)
        pareto_results.append({
            'distribution': 'Pareto',
            'replicate_size': n,
            'ks_statistic': ks_stat,
            'p_value': p_value
        })
    
    plot_ks_comparison(replicate_sizes, uniform_ks_stats, pareto_ks_stats,
                      'ks_test_comparison.png')
    
    all_results = uniform_results + pareto_results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('../datasets/ks_test_results.csv', index=False)

if __name__ == "__main__": main()
