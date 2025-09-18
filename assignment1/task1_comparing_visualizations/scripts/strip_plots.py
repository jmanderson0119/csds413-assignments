import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import matplotlib.ticker as mtick

def strip_plot_albums_data(input_csv: str, x: str, y: str):
    """
    Generates a strip plot for album sales by genre.

    :param input_csv: Path to album_sales by genre data
    :type input_csv: str
    :param x: Attribute designated x-col in input_csv
    :type x: str
    :param y: Attribute designated y-col in input_csv
    :type y: str
    """
    df = pd.read_csv(input_csv)

    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(12, 7))

    ax = sns.stripplot(data=df, x=x, y=y, alpha=0.1, size=24, jitter=False, zorder=1)

    sns.pointplot(data=df, x=x, y=y, errorbar=None, estimator='mean', markers='D', markersize=8, color='darkblue', linestyles='none', zorder=2, ax=ax)

    ax.set_xticks(range(len(df[x].unique())))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=12)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    ax.set_xlabel("Genres", fontsize=14, labelpad=15)
    ax.set_ylabel("Album Sales", fontsize=14, labelpad=15)
    ax.set_title("Distribution of Album Sales by Genre", fontsize=16, pad=20)

    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig("../figures/album_sales_by_genre_strip_plot.png", bbox_inches='tight', dpi=300)
