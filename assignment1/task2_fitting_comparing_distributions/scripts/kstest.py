import pandas as pd
from scipy import stats

def kstest(real_csv: str, synthetic_csvs: list, labels: list):
    real_data = pd.read_csv(real_csv).iloc[:, 0].to_numpy()
    
    results = []
    
    for synth_csv, label in zip(synthetic_csvs, labels):
        synthetic_data = pd.read_csv(synth_csv).iloc[:, 0].to_numpy()
        
        statistic, p_value = stats.ks_2samp(real_data, synthetic_data)
        
        results.append({
            'distribution': label,
            'ks_statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    return results

print(kstest('../datasets/uniform/clean/d20_rolls.csv', ['../datasets/uniform/synth/rolls_gaussian.csv', '../datasets/uniform/synth/rolls_uniform.csv', 
        '../datasets/uniform/synth/rolls_powerlaw.csv', '../datasets/uniform/synth/rolls_exponential.csv'], ['Normal', 'Uniform', 'Power Law', 'Exponential']))