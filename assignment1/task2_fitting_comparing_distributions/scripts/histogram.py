import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def histogram(real_csv: str, synthetic_csvs: list, labels: list, output_filename: str,
              xlabel: str = 'Value', ylabel: str = 'Density', title: str = 'Histogram of Real vs Synthetic Distributions'):
    real_data = pd.read_csv(real_csv).iloc[:, 0]

    data_list = [real_data]
    label_list = ['Real Data']

    for synth_csv, label in zip(synthetic_csvs, labels):
        synthetic_data = pd.read_csv(synth_csv).iloc[:, 0]
        data_list.append(synthetic_data)
        label_list.append(label)

    all_data = pd.concat(data_list, keys=label_list).reset_index(level=0)
    all_data.columns = ['Model', 'Data']

    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(12, 7))

    ax = sns.histplot(data=all_data, x='Data', hue='Model', bins=40,
                 stat='density', alpha=0.7, edgecolor='white', multiple='layer')

    ax.set_xlabel(xlabel, fontsize=14, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=15)
    ax.set_title(title, fontsize=16, pad=20)

    if ax.get_legend() is not None:
        legend = ax.get_legend()
        for text in legend.get_texts():
            text.set_fontsize(11)

    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_filename}", bbox_inches='tight', dpi=300)

histogram('../datasets/uniform/clean/d20_rolls.csv', ['../datasets/uniform/synth/rolls_gaussian.csv', '../datasets/uniform/synth/rolls_uniform.csv'],
        #'../datasets/uniform/synth/rolls_powerlaw.csv', '../datasets/uniform/synth/rolls_exponential.csv'], 
        ['Gaussian', 'Uniform' #'Power Law', 'Exponential'
         ],
        'uniform/histogram.png', xlabel='D20 Roll Value', ylabel='Density of Rolls', title='Histogram of Real vs Synthetic D20 Roll Distributions')