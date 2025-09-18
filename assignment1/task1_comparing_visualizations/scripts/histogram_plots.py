import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import matplotlib.ticker as mtick

def histogram_plot_albums_data(input_csv: str, x: str, hue: str):
    """
    Generates a histogram of album sales grouped by genre.

    :param input_csv: Path to album_sales by genre data
    :type input_csv: str
    :param x: The numerical variable to plot
    :type x: str
    :param hue: The categorical variable to group by
    :type hue: str
    """
    df = pd.read_csv(input_csv)

    sns.set_style("whitegrid")
    sns.set_palette("ch:s=.25,rot=-.25")
    plt.figure(figsize=(12, 7))

    ax = sns.histplot(data=df, x=x, hue=hue, bins=20, kde=False, multiple="stack", 
                      edgecolor="white", alpha=1.0, legend=True)

    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    ax.set_xlabel("Album Sales", fontsize=14, labelpad=15)
    ax.set_ylabel("Number of Albums", fontsize=14, labelpad=15)
    ax.set_title("Histogram of Album Sales by Genre", fontsize=16, pad=20)

    if ax.get_legend() is not None:
        legend = ax.get_legend()
        legend.set_title("Genre")
        for text in legend.get_texts():
            text.set_fontsize(11)
        legend.get_title().set_fontsize(12)

    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig("../figures/album_sales_by_genre_histogram_plot.png", bbox_inches='tight', dpi=300)
