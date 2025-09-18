import matplotlib.pyplot as plt
import pandas as pd
import os 
import seaborn as sns
import matplotlib.ticker as mtick

def error_bar_plot_albums_data(input_csv: str, x: str, y: str):
    """
    Generates an error bar plot for album sales by genre.

    :param input_csv: Path to album_sales by genre data
    :type input_csv: str
    :param x: Attribute designated x-col in input_csv
    :type X: str
    :param y: Attribute designated y-col in input_csv
    :type y: str
    """
    df = pd.read_csv(input_csv)

    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(12, 7))
    ax = sns.pointplot(data=df, x=x, y=y, capsize=0.1, err_kws={'linewidth': 1.5}, linewidth=1.2, markers='o')

    # Improves x-axis label readability
    ax.set_xticks(range(len(df[x].unique())))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=12)

    # Formats y-axis labels in millions
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    ax.set_xlabel("Genres", fontsize=14, labelpad=15)
    ax.set_ylabel("Mean Album Sales", fontsize=14, labelpad=15)
    ax.set_title("Average Album Sales by Genre", fontsize=16, pad=20)

    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig("../figures/album_sales_by_genre_error_bar_plot.png", bbox_inches='tight', dpi=300)
