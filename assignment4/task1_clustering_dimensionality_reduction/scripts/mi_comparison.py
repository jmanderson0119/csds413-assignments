import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
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


def compute_agreement_metric(cluster_labels: np.ndarray, 
                             party_labels: np.ndarray) -> float:
    """
    Computes agreement metric between clusters and party affiliations.
    
    :param cluster_labels: Cluster assignments
    :type cluster_labels: np.ndarray
    :param party_labels: Party affiliation labels
    :type party_labels: np.ndarray
    :returns: Normalized Mutual Information
    :rtype: float
    """
    nmi = normalized_mutual_info_score(party_labels, cluster_labels)
    
    return nmi


def plot_comparison(votes_pca: np.ndarray, cluster_labels_full: np.ndarray,
                   cluster_labels_pca: np.ndarray, output_filename: str):
    """
    Plots side-by-side comparison of clustering on full data vs PCs.
    
    :param votes_pca: pca-transformed data
    :type votes_pca: np.ndarray
    :param cluster_labels_full: Cluster assignments from full data
    :type cluster_labels_full: np.ndarray
    :param cluster_labels_pca: Cluster assignments from pca data
    :type cluster_labels_pca: np.ndarray
    :param party_labels: Party affiliation labels
    :type party_labels: np.ndarray
    :param output_filename: Filename for output figure
    :type output_filename: str
    """
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    cluster_colors = ['purple', 'gold']
    
    cluster_0_full_mean_pc1 = np.mean(votes_pca[cluster_labels_full == 0, 0])
    cluster_1_full_mean_pc1 = np.mean(votes_pca[cluster_labels_full == 1, 0])
    flip_full = cluster_0_full_mean_pc1 > cluster_1_full_mean_pc1
    
    cluster_0_pca_mean_pc1 = np.mean(votes_pca[cluster_labels_pca == 0, 0])
    cluster_1_pca_mean_pc1 = np.mean(votes_pca[cluster_labels_pca == 1, 0])
    flip_pca = cluster_0_pca_mean_pc1 > cluster_1_pca_mean_pc1
    
    # Clusters on full data
    for cluster in range(2):
        mask = cluster_labels_full == cluster
        color_idx = 1 - cluster if flip_full else cluster
        ax1.scatter(votes_pca[mask, 0], votes_pca[mask, 1],
                   c=cluster_colors[color_idx], label=f'Cluster {cluster}',
                   alpha=0.6, s=50, edgecolor='white', linewidth=0.5)
    
    ax1.set_xlabel('PC1', fontsize=14, labelpad=15)
    ax1.set_ylabel('PC2', fontsize=14, labelpad=15)
    ax1.set_title('K-means on Full Data', fontsize=16, pad=20)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Clusters on pc1-pc2
    for cluster in range(2):
        mask = cluster_labels_pca == cluster
        color_idx = 1 - cluster if flip_pca else cluster
        ax2.scatter(votes_pca[mask, 0], votes_pca[mask, 1],
                   c=cluster_colors[color_idx], label=f'Cluster {cluster}',
                   alpha=0.6, s=50, edgecolor='white', linewidth=0.5)
    
    ax2.set_xlabel('PC1', fontsize=14, labelpad=15)
    ax2.set_ylabel('PC2', fontsize=14, labelpad=15)
    ax2.set_title('K-means on PC1-PC2', fontsize=16, pad=20)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_filename}", bbox_inches='tight', dpi=300)


def main():
    votes, party_labels = load_congressional_data(
        '../datasets/p1_congress_1984_votes.csv',
        '../datasets/p1_congress_1984_party_affiliations.csv')
    
    _, cluster_labels_full, inertia_full = cluster_kmeans(votes, n_clusters=2)
    nmi_full = compute_agreement_metric(cluster_labels_full, party_labels)
    
    print("Full:")
    print(f"Inertia: {inertia_full:.2f}")
    print(f"Mutual Information: {nmi_full:.4f}")
    
    # PCA transformation
    pca = PCA(n_components=2)
    votes_pca = pca.fit_transform(votes)
    
    # Clustering on PC1-PC2 only
    _, cluster_labels_pca, inertia_pca = cluster_kmeans(votes_pca, n_clusters=2)
    nmi_pca = compute_agreement_metric(cluster_labels_pca, party_labels)
    
    print("PC1-PC2:")
    print(f"Inertia: {inertia_pca:.2f}")
    print(f"Mutual Information: {nmi_pca:.4f}")
    
    plot_comparison(votes_pca, cluster_labels_full, cluster_labels_pca, 'clustering_comparison.png')


if __name__ == "__main__":
    np.random.seed(42)
    main()