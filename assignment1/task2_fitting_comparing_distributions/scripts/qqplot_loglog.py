import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

def qqplot_loglog(real_csv: str, synthetic_csvs: list, labels: list, output_filename: str):
    real_data = pd.read_csv(real_csv).iloc[:, 0].to_numpy()

    real_data = real_data[real_data > 0]

    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(12, 8))

    percentiles = np.linspace(0, 100, 75)

    all_positive_quantiles = []

    for synth_csv, label in zip(synthetic_csvs, labels):
        synthetic_data = pd.read_csv(synth_csv).iloc[:, 0].to_numpy()

        synthetic_data = synthetic_data[synthetic_data > 0]

        real_quantiles = np.percentile(real_data, percentiles)
        synthetic_quantiles = np.percentile(synthetic_data, percentiles)

        log_real_quantiles = np.log10(real_quantiles)
        log_synthetic_quantiles = np.log10(synthetic_quantiles)

        plt.scatter(log_real_quantiles, log_synthetic_quantiles, alpha=0.6, s=120, label=label)

        all_positive_quantiles.extend([real_quantiles, synthetic_quantiles])

    min_val = np.log10(min([np.min(q) for q in all_positive_quantiles if len(q) > 0]))
    max_val = np.log10(max([np.max(q) for q in all_positive_quantiles if len(q) > 0]))
    plt.plot([min_val, max_val], [min_val, max_val], 'k-', alpha=0.8)

    plt.xlabel('Log_10(Real Data Quantiles)', fontsize=14, labelpad=15)
    plt.ylabel('Log_10(Synthetic Data Quantiles)', fontsize=14, labelpad=15)
    plt.title('Log-Log Q-Q Plot of Real vs Synthetic Distributions', fontsize=16, pad=20)

    legend = plt.legend()
    for text in legend.get_texts():
        text.set_fontsize(11)

    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_filename}", bbox_inches='tight', dpi=300)
