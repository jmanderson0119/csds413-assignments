import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import matplotlib.ticker as mtick

def histogram_plot(input_csv: str, x: str, hue: str, x_label: str, y_label: str, 
                   category: str, title: str, output_csv: str):
    """
    Generates a histogram.

    :param input_csv: Path to album_sales by genre data
    :type input_csv: str
    :param x: The numerical variable to plot
    :type x: str
    :param hue: The categorical variable to group by
    :type hue: str
    :param x_label: label for the x axis
    :type x_label: str
    :param y_label: label for the y axis
    :type y_label: str
    :param category: title for the category being grouped by
    :param title: title for the figure
    :type title: str
    :param output_csv: Filename for the figure
    :type output_csv: str
    """
    df = pd.read_csv(input_csv)

    sns.set_style("whitegrid")
    sns.set_palette("ch:s=.25,rot=-.25")
    plt.figure(figsize=(12, 7))

    ax = sns.histplot(data=df, x=x, hue=hue, bins=20, kde=False, multiple="stack", 
                      edgecolor="white", alpha=1.0, legend=True)

    if df[x].max() >= 1e6: ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    ax.set_xlabel(x_label, fontsize=14, labelpad=15)
    ax.set_ylabel(y_label, fontsize=14, labelpad=15)
    ax.set_title(title, fontsize=16, pad=20)

    if ax.get_legend() is not None:
        legend = ax.get_legend()
        legend.set_title(category)
        for text in legend.get_texts():
            text.set_fontsize(11)
        legend.get_title().set_fontsize(12)

    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{output_csv}", bbox_inches='tight', dpi=300)
