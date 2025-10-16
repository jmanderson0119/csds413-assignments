import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    data = pd.read_csv('../datasets/cars.csv', header=None)
    us_mileage = data[0].dropna()
    jp_mileage = data[1].dropna()

    df = pd.DataFrame({
        'Mileage': pd.concat([us_mileage, jp_mileage], ignore_index=True),
        'Region': ['US'] * len(us_mileage) + ['Japan'] * len(jp_mileage)
    })

    sns.set_style("whitegrid")
    sns.set_palette("muted")

    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x='Region', y='Mileage', data=df, showfliers=True)

    ax.set_title('Car Mileages for US vs Japanese Cars', fontsize=16)
    ax.set_xlabel('')
    ax.set_ylabel('Mileage', fontsize=12)

    plt.tight_layout()
    plt.savefig("../figures/boxplot.png", bbox_inches='tight', dpi=300)

if __name__ == "__main__": main()
