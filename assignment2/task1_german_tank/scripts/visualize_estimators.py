import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_estimator_means(results_df: pd.DataFrame, output_dir: str):
    """
    Plots the mean of each estimator as a function of n, for each M value.
    
    :param results_df: DataFrame of estimator means and variances
    :type results_df: pd.DataFrame
    :param output_dir: Where to save the figures
    :type output_dir: str
    """
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    
    M_values = results_df['M'].unique()
    
    for M in M_values:
        df = results_df[results_df['M'] == M]
        
        plt.figure(figsize=(12, 7))
        
        ax = plt.gca()
        ax.plot(df['n'], df['mle_mean'], 
                marker='o', markersize=4, label='MLE', linewidth=2, alpha=0.8)
        ax.plot(df['n'], df['mean_mean'], 
                marker='o', markersize=4, label='MEAN', linewidth=2, alpha=0.8)
        ax.plot(df['n'], df['mvu_mean'], 
                marker='o', markersize=4, label='MVU', linewidth=2, alpha=0.8)
        ax.axhline(y=M, color='black', linestyle='--', linewidth=1.5, 
                   label=f'True M={M}')
        
        ax.set_xlabel('Sample Size n', fontsize=14, labelpad=15)
        ax.set_ylabel('Mean Estimate', fontsize=14, labelpad=15)
        ax.set_title(f'Mean of Estimators vs Sample Size (M={M})', fontsize=16, pad=20)
        
        legend = ax.legend(fontsize=12, loc='best')
        for text in legend.get_texts():
            text.set_fontsize(11)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/mean_comparison_M{M}.png', bbox_inches='tight', dpi=300)
        
def plot_estimator_variances(results_df: pd.DataFrame, output_dir: str):
    """
    Plots the variance of each estimator as a function of n, for each M value.
    
    :param results_df: DataFrame from estimator means and variances
    :type results_df: pd.DataFrame
    :param output_dir: Where to save the figures
    :type output_dir: str
    """
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    
    M_values = results_df['M'].unique()
    
    for M in M_values:
        df = results_df[results_df['M'] == M]
        
        plt.figure(figsize=(12, 7))
        
        ax = plt.gca()
        ax.plot(df['n'], df['mle_var'], 
                marker='d', markersize=8, label='MLE', linewidth=2, alpha=0.8)
        ax.plot(df['n'], df['mean_var'], 
                marker='o', markersize=4, label='MEAN', linewidth=2, alpha=0.8)
        ax.plot(df['n'], df['mvu_var'], 
                marker='o', markersize=4, label='MVU', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Sample Size n', fontsize=14, labelpad=15)
        ax.set_ylabel('Variance', fontsize=14, labelpad=15)
        ax.set_title(f'Variance of Estimators vs Sample Size (M={M})', fontsize=16, pad=20)
        
        legend = ax.legend(fontsize=12, loc='best')
        for text in legend.get_texts():
            text.set_fontsize(11)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/variance_comparison_M{M}.png', bbox_inches='tight', dpi=300)
