import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os


def load_congressional_data(votes_file: str, party_file: str) -> tuple:
    """
    Loads congressional votes and party affiliation data.
    
    :param votes_file: Path to votes CSV file
    :type votes_file: str
    :param party_file: Path to party affiliations CSV file
    :type party_file: str
    :returns: Tuple of (votes array, party labels array)
    :rtype: tuple
    """
    votes_df = pd.read_csv(votes_file, header=None)
    party_df = pd.read_csv(party_file, header=None)
    
    votes = votes_df.values
    party_labels = party_df.values.flatten()
    
    return votes, party_labels


def congress_pca(data: np.ndarray, n_components: int = None) -> tuple:
    """
    Applies PCA to the congressional voting data.
    
    :param data: Feature matrix
    :type data: np.ndarray
    :param n_components: Number of components to keep
    :type n_components: int or None
    :returns: PCA model, transformed data, explained variance ratio
    :rtype: tuple
    """
    
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    
    return pca, data_pca, explained_variance


def plot_cumulative_variance(explained_variance: np.ndarray, 
                            output_filename: str):
    """
    Plots cumulative variance explained by top k principal components.
    
    :param explained_variance: Array of explained variance ratios
    :type explained_variance: np.ndarray
    :param output_filename: Filename for output figure
    :type output_filename: str
    """
    cumulative_variance = np.cumsum(explained_variance)
    k_values = np.arange(1, len(cumulative_variance) + 1)
    
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(10, 6))
    
    plt.plot(k_values, cumulative_variance, marker='o', 
             linewidth=2, markersize=6, color='steelblue')
    
    plt.xlabel('Number of Principal Components (k)', fontsize=14, labelpad=15)
    plt.ylabel('Cumulative Variance Explained', fontsize=14, labelpad=15)
    plt.title('Cumulative Variance Explained by Top k Principal Components', 
             fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, len(k_values) + 0.5)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_filename}", bbox_inches='tight', dpi=300)


def plot_pc_scatter(data_pca: np.ndarray, party_labels: np.ndarray,
                   pc_x: int, pc_y: int, output_filename: str):
    """
    Plots scatter plot of two principal components colored by party.
    
    :param data_pca: pca-transformed data
    :type data_pca: np.ndarray
    :param party_labels: Party affiliation labels
    :type party_labels: np.ndarray
    :param pc_x: Index of PC for x-axis
    :type pc_x: int
    :param pc_y: Index of PC for y-axis
    :type pc_y: int
    :param output_filename: Filename for output figure
    :type output_filename: str
    """
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(10, 6))
    
    # Creates color mapping for parties
    unique_parties = np.unique(party_labels)
    colors = {'republican': 'red', 'democrat': 'blue'}
    
    for party in unique_parties:
        mask = party_labels == party
        party_lower = party.lower()
        color = colors.get(party_lower, 'gray')
        
        plt.scatter(data_pca[mask, pc_x], data_pca[mask, pc_y],
                   c=color, label=party.capitalize(), 
                   alpha=0.6, s=50, edgecolor='white', linewidth=0.5)
    
    plt.xlabel(f'PC{pc_x + 1}', fontsize=14, labelpad=15)
    plt.ylabel(f'PC{pc_y + 1}', fontsize=14, labelpad=15)
    plt.title(f'Congress Voting Separation on PC{pc_x + 1} vs PC{pc_y + 1}', 
             fontsize=16, pad=20)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_filename}", bbox_inches='tight', dpi=300)


def main():
    votes, party_labels = load_congressional_data(
        '../datasets/p1_congress_1984_votes.csv',
        '../datasets/p1_congress_1984_party_affiliations.csv')
    
    _, n_votes = votes.shape
    _, votes_pca, explained_variance = congress_pca(votes, n_components=n_votes)
    
    plot_cumulative_variance(explained_variance, 'cumulative_variance_explained.png')
    
    # Reports variance explained for a few numbers of pcs
    cumulative_variance = np.cumsum(explained_variance)
    for k in [1, 2, 3, 4, 5, 10]:
        if k <= len(cumulative_variance): print(f"k={k}: {cumulative_variance[k-1]:.4f}")
    
    # Finds number of pcs for a few thresholds for discussion
    for threshold in [0.80, 0.90, 0.95]:
        k_needed = np.argmax(cumulative_variance >= threshold) + 1
        print(f"{threshold:.0%} variance: {k_needed}")
    
    # Project onto first 3 pcs and create scatter plots
    pc_pairs = [(0, 1), (0, 2), (1, 2)]
    
    for pc_x, pc_y in pc_pairs:
        output_file = f'scatter_pc{pc_x+1}_pc{pc_y+1}.png'
        plot_pc_scatter(votes_pca, party_labels, pc_x, pc_y, output_file)
    
    for i in range(3):
        print(f"PC{i+1}: {explained_variance[i]:.4f}")

if __name__ == "__main__": 
    np.random.seed(42)
    main()
