import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    data = pd.read_csv('../datasets/cars.csv', header=None)
    us_mileage = data[0].dropna()
    jp_mileage = data[1].dropna()

    sns.set_style("whitegrid")
    sns.set_palette("muted")

    plt.figure(figsize=(10, 6))

    sns.histplot(us_mileage, bins=30, kde=False, color='blue', edgecolor='white', 
    alpha=0.6, label='US Cars')
    sns.histplot(jp_mileage, bins=30, kde=False, color='red', edgecolor='white', 
    alpha=0.6, label='Japanese Cars')

    plt.title('US vs Japanese Cars Mileage Distributions', fontsize=16)
    plt.xlabel('Mileage', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(fontsize=12)

    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig("../figures/histogram.png", bbox_inches='tight', dpi=300)

if __name__ == "__main__": main()
