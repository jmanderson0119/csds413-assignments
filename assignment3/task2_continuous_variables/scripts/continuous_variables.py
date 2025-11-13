import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

def compute_pearson_correlation(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Computes Pearson correlation coefficient and p-value.
    
    :param x: First variable
    :type x: np.ndarray
    :param y: Second variable
    :type y: np.ndarray
    :returns: Tuple of correlation coefficient, p-value
    :rtype: tuple
    """
    r, p_value = pearsonr(x, y)
    return r, p_value

def plot_scatter(x: np.ndarray, y: np.ndarray, dataset_name: str, 
                r: float, p_value: float, output_filename: str):
    """
    Creates scatter plot for two continuous variables.
    
    :param x: First variable
    :type x: np.ndarray
    :param y: Second variable
    :type y: np.ndarray
    :param dataset_name: Name of the dataset
    :type dataset_name: str
    :param r: Pearson correlation coefficient
    :type r: float
    :param p_value: P-value for correlation
    :type p_value: float
    :param output_filename: Filename for output figure
    :type output_filename: str
    """
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(10, 6))
    
    plt.scatter(x, y, alpha=0.6, s=30, color='steelblue')
    
    plt.xlabel('Variable 1', fontsize=14, labelpad=15)
    plt.ylabel('Variable 2', fontsize=14, labelpad=15)
    plt.title(f'Scatter Plot: {dataset_name}', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    
    if p_value < 0.0001: p_text = f'p-value = {p_value:.2e}'
    else: p_text = f'p-value = {p_value:.6f}'
    
    plt.text(0.1, 0.95, f'r = {r:.4f}\n{p_text}',
            transform=plt.gca().transAxes, verticalalignment='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_filename}", bbox_inches='tight', dpi=300)

def analyze_dataset(data_file: str, dataset_name: str, 
                   output_filename: str, alpha: float = 0.05):
    """
    Analyzes correlation for a single dataset.
    
    :param data_file: Path to CSV file
    :type data_file: str
    :param dataset_name: Name of the dataset
    :type dataset_name: str
    :param output_filename: Filename for output figure
    :type output_filename: str
    :param alpha: Significance level
    :type alpha: float
    :returns: Dictionary with results
    :rtype: dict
    """
    data = pd.read_csv(data_file, header=None)
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    
    r, p_value = compute_pearson_correlation(x, y)
    
    plot_scatter(x, y, dataset_name, r, p_value, output_filename)
    
    return {
        'n_samples': len(x),
        'r': r,
        'p_value': p_value,
        'significant': p_value < alpha
    }

def analyze_parta(alpha: float = 0.05):
    """
    Performs analysis for Part A.
    
    :param alpha: Significance level
    :type alpha: float
    """
    results = analyze_dataset('../datasets/p2a.csv', 'Dataset A (p2a.csv)', 
                             'scatter_p2a.png', alpha)
    
    print(f"n={results['n_samples']}, r={results['r']:.6f}, p={results['p_value']:.6f}")
    
    return results

def analyze_partb(alpha: float = 0.05):
    """
    Performs analysis for Part B.
    
    :param alpha: Significance level
    :type alpha: float
    """
    results_a = analyze_dataset('../datasets/p2a.csv', 'Dataset A (p2a.csv)', 
                                'scatter_p2a.png', alpha)
    
    results_b = analyze_dataset('../datasets/p2b.csv', 'Dataset B (p2b.csv)', 
                                'scatter_p2b.png', alpha)
    
    print(f"n={results_a['n_samples']}, r={results_a['r']:.6f}, p={results_a['p_value']:.6f}")
    print(f"n={results_b['n_samples']}, r={results_b['r']:.6f}, p={results_b['p_value']:.6f}")
    
    return results_a, results_b

def analyze_partc(alpha: float = 0.05):
    """
    Performs analysis for Part C.
    
    :param alpha: Significance level
    :type alpha: float
    """
    results_a = analyze_dataset('../datasets/p2a.csv', 'Dataset A (p2a.csv)', 
                                'scatter_p2a.png', alpha)
    
    results_c = analyze_dataset('../datasets/p2c.csv', 'Dataset C (p2c.csv)', 
                                'scatter_p2c.png', alpha)
    
    print(f"n={results_a['n_samples']}, r={results_a['r']:.6f}, p={results_a['p_value']:.6f}")
    print(f"n={results_c['n_samples']}, r={results_c['r']:.6f}, p={results_c['p_value']:.6f}")
    
    return results_a, results_c

def main():
    np.random.seed(42)
    
    analyze_parta(alpha=0.05)
    analyze_partb(alpha=0.05)
    analyze_partc(alpha=0.05)

if __name__ == "__main__": main()