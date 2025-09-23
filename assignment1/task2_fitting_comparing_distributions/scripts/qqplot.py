import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

def qqplot(real_csv: str, synthetic_csvs: list, labels: list, output_filename: str):
    real_data = pd.read_csv(real_csv).iloc[:, 0].to_numpy()
    
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(12, 8))
    
    percentiles = np.linspace(0, 100, 75)
    
    for synth_csv, label in zip(synthetic_csvs, labels):
        synthetic_data = pd.read_csv(synth_csv).iloc[:, 0].to_numpy()
        
        real_quantiles = np.percentile(real_data, percentiles)
        synthetic_quantiles = np.percentile(synthetic_data, percentiles)
        
        plt.scatter(real_quantiles, synthetic_quantiles, alpha=0.6, s=120, label=label)
    
    min_val = min(np.min(real_data), min([np.min(pd.read_csv(csv).iloc[:, 0]) for csv in synthetic_csvs]))
    max_val = max(np.max(real_data), max([np.max(pd.read_csv(csv).iloc[:, 0]) for csv in synthetic_csvs]))
    plt.plot([min_val, max_val], [min_val, max_val], 'k-', alpha=0.8)
    
    plt.xlabel('Real Data Quantiles', fontsize=14, labelpad=15)
    plt.ylabel('Synthetic Data Quantiles', fontsize=14, labelpad=15)
    plt.title('Q-Q Plot of Real vs Synthetic Distributions', fontsize=16, pad=20)
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_fontsize(11)
    
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_filename}", bbox_inches='tight', dpi=300)
