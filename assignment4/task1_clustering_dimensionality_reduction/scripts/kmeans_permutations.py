import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
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


def cluster_kmeans(data: np.ndarray, n_clusters: int = 2, 
                   random_state: int = 42) -> tuple:
    """
    Applies K-means clustering to the data.
    
    :param data: Feature matrix
    :type data: np.ndarray
    :param n_clusters: Number of clusters
    :type n_clusters: int
    :param random_state: Random seed
    :type random_state: int
    :returns: KMeans model, cluster labels, inertia score
    :rtype: tuple
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    inertia = kmeans.inertia_
    
    return kmeans, cluster_labels, inertia


def plot_clusters_on_pcs(data_pca: np.ndarray, cluster_labels: np.ndarray,
                         party_labels: np.ndarray, output_filename: str):
    """
    Plots clusters on first two principal components with party comparison.
    
    :param data_pca: pca-transformed data
    :type data_pca: np.ndarray
    :param cluster_labels: Cluster assignments
    :type cluster_labels: np.ndarray
    :param party_labels: Party affiliation labels
    :type party_labels: np.ndarray
    :param output_filename: Filename for output figure
    :type output_filename: str
    """
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Determine which cluster corresponds to which party (majority vote)
    cluster_0_mask = cluster_labels == 0
    cluster_0_parties = party_labels[cluster_0_mask]
    cluster_0_is_republican = np.sum(cluster_0_parties == 'Republican') > np.sum(cluster_0_parties == 'Democrat')
    
    # Left plot: colored by cluster, outlined in red if misclassified
    cluster_colors = ['purple', 'gold']
    
    for i in range(len(data_pca)):
        cluster = cluster_labels[i]
        party = party_labels[i]
        
        # Check if misclassified
        if cluster == 0:
            expected_party = 'Republican' if cluster_0_is_republican else 'Democrat'
        else:
            expected_party = 'Democrat' if cluster_0_is_republican else 'Republican'
        
        is_misclassified = (party != expected_party)
        edge_color = 'red' if is_misclassified else 'white'
        edge_width = 1.5 if is_misclassified else 0.5
        
        ax1.scatter(data_pca[i, 0], data_pca[i, 1],
                   c=cluster_colors[cluster], alpha=0.6, s=50,
                   edgecolor=edge_color, linewidth=edge_width)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
               markersize=6, label='Cluster 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', 
               markersize=6, label='Cluster 1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markeredgecolor='red', markersize=6, linewidth=1.5, label='Misclassified')
    ]
    
    ax1.set_xlabel('PC1', fontsize=14, labelpad=15)
    ax1.set_ylabel('PC2', fontsize=14, labelpad=15)
    ax1.set_title('K-means Clusters', fontsize=16, pad=20)
    ax1.legend(handles=legend_elements, fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: by party only
    colors = {'republican': 'red', 'democrat': 'blue'}
    for party in np.unique(party_labels):
        mask = party_labels == party
        color = colors.get(party.lower(), 'gray')
        ax2.scatter(data_pca[mask, 0], data_pca[mask, 1],
                   c=color, label=party.capitalize(),
                   alpha=0.6, s=50, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('PC1', fontsize=14, labelpad=15)
    ax2.set_ylabel('PC2', fontsize=14, labelpad=15)
    ax2.set_title('Congress Voting Separation on PC1 vs PC2', fontsize=16, pad=20)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_filename}", bbox_inches='tight', dpi=300)


def permute_votes(votes: np.ndarray) -> np.ndarray:
    """
    Permutes votes randomly across issues for each congress member.
    
    :param votes: Original votes matrix
    :type votes: np.ndarray
    :returns: Permuted votes matrix
    :rtype: np.ndarray
    """
    permuted = votes.copy()
    for i in range(votes.shape[0]):
        permuted[i, :] = np.random.permutation(votes[i, :])
    
    return permuted


def permutation_test_clustering(votes: np.ndarray, n_permutations: int = 1000,
                                n_clusters: int = 2) -> tuple:
    """
    Performs permutation test for clustering significance.
    
    :param votes: Original votes matrix
    :type votes: np.ndarray
    :param n_permutations: Number of permutations
    :type n_permutations: int
    :param n_clusters: Number of clusters
    :type n_clusters: int
    :returns: Original inertia, null distribution, p-value
    :rtype: tuple
    """
    # Cluster original data
    _, _, original_inertia = cluster_kmeans(votes, n_clusters=n_clusters)
    
    # Generates null distribution
    null_inertias = np.zeros(n_permutations)
    for i in range(n_permutations):
        permuted_votes = permute_votes(votes)
        _, _, null_inertias[i] = cluster_kmeans(permuted_votes, 
                                               n_clusters=n_clusters,
                                               random_state=i)
    
    p_value = (np.sum(null_inertias <= original_inertia) + 1) / (n_permutations + 1)
    
    return original_inertia, null_inertias, p_value


def plot_permutation_results(original_inertia: float, null_inertias: np.ndarray,
                             p_value: float, output_filename: str):
    """
    Plots permutation test results.
    
    :param original_inertia: Inertia from original data
    :type original_inertia: float
    :param null_inertias: Null distribution of inertias
    :type null_inertias: np.ndarray
    :param p_value: Computed p-value
    :type p_value: float
    :param output_filename: Filename for output figure
    :type output_filename: str
    """
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(10, 6))
    
    plt.hist(null_inertias, bins=30, color='steelblue', 
             alpha=0.7, edgecolor='white')
    
    plt.text(0.95, 0.95, f'Observed: {original_inertia:.2f} (p = {p_value:.3f})',
            transform=plt.gca().transAxes, verticalalignment='top',
            horizontalalignment='right', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel('K-means Inertia', fontsize=14, labelpad=15)
    plt.ylabel('Frequency', fontsize=14, labelpad=15)
    plt.title('Distribution of K-Means Inertia for 1000 Permutations', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_filename}", bbox_inches='tight', dpi=300)


def main():
    votes, party_labels = load_congressional_data(
        '../datasets/p1_congress_1984_votes.csv',
        '../datasets/p1_congress_1984_party_affiliations.csv')
    
    pca = PCA(n_components=2)
    votes_pca = pca.fit_transform(votes)

    _, cluster_labels, inertia = cluster_kmeans(votes, n_clusters=2)
    print(f"Inertia: {inertia:.2f}")
    
    plot_clusters_on_pcs(votes_pca, cluster_labels, party_labels, 
                        'clusters_vs_parties.png')
    
    original_inertia, null_inertias, p_value = permutation_test_clustering(
        votes, n_permutations=1000, n_clusters=2)
    
    print(f"Original inertia: {original_inertia:.2f}")
    print(f"Mean null inertia: {np.mean(null_inertias):.2f}")
    print(f"p-value: {p_value:.4f}")
    
    plot_permutation_results(original_inertia, null_inertias, p_value,
                           'permutation_test_clustering.png')


if __name__ == "__main__":
    np.random.seed(42)
    main()
