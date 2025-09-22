import matplotlib.pyplot as plt
import pandas as pd
import os 
import seaborn as sns
import matplotlib.ticker as mtick

def error_bar_plot(input_csv: str, x: str, y: str, x_label: str, y_label: str, 
                   title: str, output_csv: str, hue: str=None):
    """
    Generates an error bar plot.

    :param input_csv: Path to input dataset
    :type input_csv: str
    :param x: Attribute designated x-col in input_csv
    :type X: str
    :param y: Attribute designated y-col in input_csv
    :type y: str
    :param x_label: label for the x axis
    :type x_label: str
    :param y_label: label for the y axis
    :type y_label: str
    :param title: title for the figure
    :type title: str
    :param output_csv: Filename for the figure
    :type output_csv: str
    :param hue: Categorical/grouping att in the dataset
    :type hue: str
    """
    df = pd.read_csv(input_csv)

    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(12, 7))
    ax = sns.pointplot(data=df, x=x, y=y, hue=hue, capsize=0.1, err_kws={'linewidth': 1.5}, linewidth=1.2, markers='o')

    # Improves x-axis label readability
    ax.set_xticks(range(len(df[x].unique())))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right', fontsize=12)

    # Formats y-axis labels in millions
    if df[y].max() >= 1e6: ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    ax.set_xlabel(x_label, fontsize=14, labelpad=15)
    ax.set_ylabel(y_label, fontsize=14, labelpad=15)
    ax.set_title(title, fontsize=16, pad=20)

    if hue: ax.legend(title=hue.capitalize(), loc='best')

    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_csv}", bbox_inches='tight', dpi=300)
