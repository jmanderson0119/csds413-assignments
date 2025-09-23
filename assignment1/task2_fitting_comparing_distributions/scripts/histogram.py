import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def histogram(real_csv: str, synthetic_csvs: list, labels: list, output_filename: str):
    real_data = pd.read_csv(real_csv).iloc[:, 0]
    
    data_list = [real_data]
    label_list = ['Real Data']
    
    for synth_csv, label in zip(synthetic_csvs, labels):
        synthetic_data = pd.read_csv(synth_csv).iloc[:, 0]
        data_list.append(synthetic_data)
        label_list.append(label)
    
    all_data = pd.concat(data_list, keys=label_list).reset_index(level=0)
    all_data.columns = ['Model', 'Sepal Width (cm)']
    
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(12, 7))
    
    ax = sns.histplot(data=all_data, x='Sepal Width (cm)', hue='Model', bins=48, 
                 stat='density', alpha=1.0, edgecolor='white', multiple='stack')
    
    ax.set_xlabel('Sepal Width (cm)', fontsize=14, labelpad=15)
    ax.set_ylabel('Density of Sepal Widths', fontsize=14, labelpad=15)
    ax.set_title('Histogram of Real vs Synthetic Distributions', fontsize=16, pad=20)
    
    if ax.get_legend() is not None:
        legend = ax.get_legend()
        for text in legend.get_texts():
            text.set_fontsize(11)
    
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_filename}", bbox_inches='tight', dpi=300)

histogram('../datasets/normal/clean/sepal_widths.csv', ['../datasets/normal/synth/sepal_widths_setosa_gaussian.csv', '../datasets/normal/synth/sepal_widths_setosa_uniform.csv', 
        '../datasets/normal/synth/sepal_widths_setosa_powerlaw.csv', '../datasets/normal/synth/sepal_widths_setosa_exponential.csv'], ['Normal', 'Uniform', 'Power Law', 'Exponential'], 
        '../figures/normal/histogram.png')