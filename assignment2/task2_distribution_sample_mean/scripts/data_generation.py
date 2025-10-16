import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_uniform_replicates(low: float, high: float, sample_size: int, 
                                num_replicates: int) -> np.ndarray:
    """
    Generates the replicates of uniformly distributed data.
    
    :param low: Lower bound of uniform distribution
    :type low: float
    :param high: Upper bound of uniform distribution
    :type high: float
    :param sample_size: Number of samples per replicate
    :type sample_size: int
    :param num_replicates: Number of replicates to generate
    :type num_replicates: int
    :returns: Array containing the replicates.
    :rtype: np.ndarray
    """
    replicates = np.random.uniform(low=low, high=high, 
                                   size=(num_replicates, sample_size))
    return replicates

def generate_powerlaw_replicates(sample_size: int, 
                                 num_replicates: int) -> np.ndarray:
    """
    Generates the replicates of power-law distributed data using Pareto distribution.
    
    :param alpha: Shape parameter for the power-law distribution
    :type alpha: float
    :param sample_size: Number of samples per replicate
    :type sample_size: int
    :param num_replicates: Number of replicates to generate
    :type num_replicates: int
    :returns: Array containing the replicates.
    :rtype: np.ndarray
    """
    replicates = np.random.pareto(a=2.3, size=(num_replicates, sample_size))
    return replicates

def plot_single_replicate(data: np.ndarray, dist_name: str, 
                                       output_filename: str):
    """
    Plots the distribution of a single replicate.
    
    :param data: Single replicate samples
    :type data: np.ndarray
    :param dist_name: Name of the distribution
    :type dist_name: str
    :param output_filename: Filename for the figure
    :type output_filename: str
    """
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(10, 6))
    
    ax = sns.histplot(data, bins=50, stat='density', 
                     edgecolor='white', alpha=0.7)
    
    ax.set_xlabel('Value', fontsize=14, labelpad=15)
    ax.set_ylabel('Density', fontsize=14, labelpad=15)
    ax.set_title(f'Distribution of {dist_name} Replicate', 
                fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(f"../figures/{output_filename}", bbox_inches='tight', dpi=300)

def save_replicates_to_csv(replicates: np.ndarray, dist_type: str, sample_size: int):
    """
    Saves generated replicates to CSV file.
    
    :param replicates: Array of replicates
    :type replicates: np.ndarray
    :param dist_type: Type of distribution
    :type dist_type: str
    :param sample_size: Number of samples per replicate
    :type sample_size: int
    """    
    df = pd.DataFrame(replicates.T)
    df.columns = [f'replicate_{i+1}' for i in range(replicates.shape[0])]
    
    filename = f'../datasets/{dist_type}_replicates_n{sample_size}.csv'
    df.to_csv(filename, index=False)

def main():
    np.random.seed(42)

    ureplicates10 = generate_uniform_replicates(low=0.25, high=1.25, 
                                               sample_size=100, num_replicates=10)
    ureplicates100 = generate_uniform_replicates(low=0.25, high=1.25, 
                                                sample_size=100, num_replicates=100)
    ureplicates1000 = generate_uniform_replicates(low=0.25, high=1.25, 
                                                 sample_size=100, num_replicates=1000)
        
    save_replicates_to_csv(ureplicates10, 'uniform', 10)
    save_replicates_to_csv(ureplicates100, 'uniform', 100)
    save_replicates_to_csv(ureplicates1000, 'uniform', 1000)
        
    plot_single_replicate(ureplicates10[0], f'Uniform(0.25, 1.25)', 
                                       'uniform_replicate.png')
    
    preplicates10 = generate_powerlaw_replicates(alpha=2.3, sample_size=100, 
                                                 num_replicates=10)
    preplicates100 = generate_powerlaw_replicates(alpha=2.3, sample_size=100, 
                                                  num_replicates=100)
    preplicates1000 = generate_powerlaw_replicates(alpha=2.3, sample_size=100, 
                                                   num_replicates=1000)
        
    save_replicates_to_csv(preplicates10, 'powerlaw', 10)
    save_replicates_to_csv(preplicates100, 'powerlaw', 100)
    save_replicates_to_csv(preplicates1000, 'powerlaw', 1000)
    
    plot_single_replicate(preplicates10[0], f'Pareto (alpha=2.3)', 
                                       'power_replicate.png')
    
if __name__ == "__main__": main()
