import pandas as pd
import numpy as np
from scipy import stats

def main():
    data = pd.read_csv('../datasets/cars.csv', header=None)
    
    us_mileage = data[0].dropna().to_numpy()
    jp_mileage = data[1].dropna().to_numpy()

    n_us = len(us_mileage)
    n_jp = len(jp_mileage)
    
    mean_us = np.mean(us_mileage)
    mean_jp = np.mean(jp_mileage)
    
    var_us = np.var(us_mileage, ddof=1)
    var_jp = np.var(jp_mileage, ddof=1)
    se = np.sqrt(var_us/n_us + var_jp/n_jp)
    
    t_stat = (mean_jp - mean_us) / se
    df = n_us + n_jp - 2
    p_value = stats.t.sf(t_stat, df)
    
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4e}")
    
if __name__ == "__main__": main()
