import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from itertools import combinations
import os

def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates mutual information between two binary variables.
    
    :param x: First binary variable
    :type x: np.ndarray
    :param y: Second binary variable
    :type y: np.ndarray
    :returns: Mutual information value
    :rtype: float
    """
    contingency = contingency_table(x, y)
    n = len(x)
    p_joint = contingency / n
    
    p_x = p_joint.sum(axis=1)
    p_y = p_joint.sum(axis=0)
    
    mi = 0
    for i in range(2):
        for j in range(2):
            if p_joint[i, j] > 0:
                mi += p_joint[i, j] * np.log2(p_joint[i, j] / (p_x[i] * p_y[j]))
    
    return mi

def jaccard_index(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates Jaccard Index between two binary variables.
    
    :param x: First binary variable
    :type x: np.ndarray
    :param y: Second binary variable
    :type y: np.ndarray
    :returns: Jaccard Index value
    :rtype: float
    """
    intersection = np.sum((x == 1) & (y == 1))
    union = np.sum((x == 1) | (y == 1))
    
    if union == 0: return 0
    
    return intersection / union

def chi_squared_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates Pearson's chi-squared statistic.
    
    :param x: First binary variable
    :type x: np.ndarray
    :param y: Second binary variable
    :type y: np.ndarray
    :returns: Chi-squared statistic
    :rtype: float
    """
    contingency = contingency_table(x, y)
    chi2_stat, _, _, _ = chi2_contingency(contingency)
    return chi2_stat

def contingency_table(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Creates 2x2 contingency table for two binary variables.
    
    :param x: First binary variable
    :type x: np.ndarray
    :param y: Second binary variable
    :type y: np.ndarray
    :returns: 2x2 contingency table
    :rtype: np.ndarray
    """
    table = np.zeros((2, 2))
    table[0, 0] = np.sum((x == 0) & (y == 0))
    table[0, 1] = np.sum((x == 0) & (y == 1))
    table[1, 0] = np.sum((x == 1) & (y == 0))
    table[1, 1] = np.sum((x == 1) & (y == 1))
    return table

def permutation_test(x: np.ndarray, y: np.ndarray, statistic_func, 
                    n_permutations: int) -> tuple:
    """
    Performs permutation test to calculate p-value for given statistic.
    
    :param x: First binary variable
    :type x: np.ndarray
    :param y: Second binary variable
    :type y: np.ndarray
    :param statistic_func: Function to calculate test statistic
    :type statistic_func: function
    :param n_permutations: Number of permutations
    :type n_permutations: int
    :returns: Tuple of observed statistic, p-value, null distribution
    :rtype: tuple
    """
    observed_stat = statistic_func(x, y)
    
    null_dist = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        y_permuted = np.random.permutation(y)
        null_dist[i] = statistic_func(x, y_permuted)
    
    c = np.sum(null_dist >= observed_stat)
    p_value = (c + 1) / (n_permutations + 1)
    
    return observed_stat, p_value, null_dist

def chi_squared_parametric(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Performs Pearson's chi-squared test using parametric distribution.
    
    :param x: First binary variable
    :type x: np.ndarray
    :param y: Second binary variable
    :type y: np.ndarray
    :returns: Tuple of chi-squared statistic, p-value
    :rtype: tuple
    """
    contingency = contingency_table(x, y)
    chi2_stat, p_value, _, _ = chi2_contingency(contingency)
    
    return chi2_stat, p_value

def plot_permutation_results(null_dist: np.ndarray, observed_stat: float, 
                            statistic_name: str, output_filename: str):
    """
    Plots sorted scatter plot of permutation test results.
    
    :param null_dist: Null distribution from permutations
    :type null_dist: np.ndarray
    :param observed_stat: Observed test statistic
    :type observed_stat: float
    :param statistic_name: Name of the statistic
    :type statistic_name: str
    :param output_filename: Filename for output figure
    :type output_filename: str
    """
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(10, 6))
    
    sorted_null = np.sort(null_dist)
    
    plt.scatter(range(len(sorted_null)), sorted_null, 
               alpha=0.5, s=10, color='steelblue', label='Null distribution')
    
    plt.axhline(y=observed_stat, color='red', linestyle='--', 
               linewidth=2, label=f'Observed value = {observed_stat:.4f}')
    
    plt.xlabel('Permutation Index (sorted)', fontsize=14, labelpad=15)
    plt.ylabel(f'{statistic_name}', fontsize=14, labelpad=15)
    plt.title(f'Permutation Test Results: {statistic_name}', 
             fontsize=16, pad=20)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_filename}", bbox_inches='tight', dpi=300)

def analyze_parta(data_file: str, n_permutations: int = 10000, 
                 alpha: float = 0.05):
    """
    Performs analysis for Part A on two genomic variants.
    
    :param data_file: Path to p1a.csv
    :type data_file: str
    :param n_permutations: Number of permutations
    :type n_permutations: int
    :param alpha: Significance level
    :type alpha: float
    """
    data = pd.read_csv(data_file, header=None)
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    
    print(f"\nn={len(x)}, N={n_permutations}, α={alpha}")
    
    mi_stat, mi_pval, mi_null = permutation_test(x, y, mutual_information, 
                                                 n_permutations)
    print(f"MI: {mi_stat:.6f}, p={mi_pval:.6f}")
    plot_permutation_results(mi_null, mi_stat, 'Mutual Information',
                           'mi_permutation_test.png')
    
    ji_stat, ji_pval, ji_null = permutation_test(x, y, jaccard_index, 
                                                 n_permutations)
    print(f"JI: {ji_stat:.6f}, p={ji_pval:.6f}")
    plot_permutation_results(ji_null, ji_stat, 'Jaccard Index',
                           'ji_permutation_test.png')
    
    chi2_stat, chi2_pval = chi_squared_parametric(x, y)
    print(f"chi^2: {chi2_stat:.6f}, p={chi2_pval:.6f}")
    
def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> tuple:
    """
    Applies Benjamini-Hochberg procedure for FDR control.
    
    :param p_values: Array of p-values
    :type p_values: np.ndarray
    :param alpha: Desired FDR level
    :type alpha: float
    :returns: Tuple of rejected hypotheses array, threshold
    :rtype: tuple
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    
    rejected_sorted = np.zeros(n, dtype=bool)
    threshold_idx = -1
    
    for k in range(n-1, -1, -1):
        if sorted_pvals[k] <= ((k + 1) / n) * alpha:
            threshold_idx = k
            break
    
    if threshold_idx >= 0:
        rejected_sorted[:threshold_idx + 1] = True
        threshold = ((threshold_idx + 1) / n) * alpha
    else:
        threshold = 0
    
    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_indices] = rejected_sorted
    
    return rejected, threshold

def analyze_partb(data_file: str, n_permutations: int = 50000, 
                 alpha: float = 0.05):
    """
    Performs analysis for Part B on multiple genomic variant pairs.
    
    :param data_file: Path to p1b.csv
    :type data_file: str
    :param n_permutations: Number of permutations
    :type n_permutations: int
    :param alpha: Significance level
    :type alpha: float
    """
    data = pd.read_csv(data_file, header=None)
    n_samples, n_vars = data.shape
    
    pairs = list(combinations(range(n_vars), 2))
    n_pairs = len(pairs)
    
    print(f"\nn={n_samples}, vars={n_vars}, pairs={n_pairs}, N={n_permutations}, α={alpha}")
    
    mi_stats = np.zeros(n_pairs)
    mi_pvals = np.zeros(n_pairs)
    ji_stats = np.zeros(n_pairs)
    ji_pvals = np.zeros(n_pairs)
    chi2_stats = np.zeros(n_pairs)
    chi2_pvals = np.zeros(n_pairs)
    
    for idx, (i, j) in enumerate(pairs):
        x = data.iloc[:, i].values
        y = data.iloc[:, j].values
        
        mi_stats[idx], mi_pvals[idx], _ = permutation_test(x, y, mutual_information, 
                                                           n_permutations)
        
        ji_stats[idx], ji_pvals[idx], _ = permutation_test(x, y, jaccard_index, 
                                                           n_permutations)
        
        chi2_stats[idx], chi2_pvals[idx] = chi_squared_parametric(x, y)
    
    mi_rejected, _ = benjamini_hochberg(mi_pvals, alpha)
    ji_rejected, _ = benjamini_hochberg(ji_pvals, alpha)
    chi2_rejected, _ = benjamini_hochberg(chi2_pvals, alpha)
    
    n_mi_sig = np.sum(mi_rejected)
    n_ji_sig = np.sum(ji_rejected)
    n_chi2_sig = np.sum(chi2_rejected)
    
    print(f"MI={n_mi_sig}, JI={n_ji_sig}, chi^2={n_chi2_sig}")
    
    mi_ji_overlap = np.sum(mi_rejected & ji_rejected)
    mi_chi2_overlap = np.sum(mi_rejected & chi2_rejected)
    ji_chi2_overlap = np.sum(ji_rejected & chi2_rejected)
    all_three_overlap = np.sum(mi_rejected & ji_rejected & chi2_rejected)
    
    print(f"MI-JI={mi_ji_overlap}, MI-chi^2={mi_chi2_overlap}, JI-chi^2={ji_chi2_overlap}, All={all_three_overlap}")
    
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].scatter(mi_stats, ji_stats, alpha=0.6, s=30, color='steelblue')
    axes[0].set_xlabel('Mutual Information', fontsize=12, labelpad=15)
    axes[0].set_ylabel('Jaccard Index', fontsize=12, labelpad=15)
    axes[0].set_title('MI vs JI', fontsize=14, pad=20)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(mi_stats, chi2_stats, alpha=0.6, s=30, color='orange')
    axes[1].set_xlabel('Mutual Information', fontsize=12, labelpad=15)
    axes[1].set_ylabel('Chi-squared Statistic', fontsize=12, labelpad=15)
    axes[1].set_title('MI vs Chi-squared', fontsize=14, pad=20)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(ji_stats, chi2_stats, alpha=0.6, s=30, color='green')
    axes[2].set_xlabel('Jaccard Index', fontsize=12, labelpad=15)
    axes[2].set_ylabel('Chi-squared Statistic', fontsize=12, labelpad=15)
    axes[2].set_title('JI vs Chi-squared', fontsize=14, pad=20)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig('../figures/statistic_comparisons.png', bbox_inches='tight', dpi=300)
   
def main():
    np.random.seed(42)
    
    analyze_parta('../datasets/p1a.csv', n_permutations=10000, alpha=0.05)
    analyze_partb('../datasets/p1b.csv', n_permutations=50000, alpha=0.05)

if __name__ == "__main__": main()
