import numpy as np
import pandas as pd
from estimators import mle_M, mean_M, mvu_M

def tank_capture(M: int, n: int) -> list:
    """
    Capture n tank IDs uniformly at random from M total tanks.
    
    :param M: Total number of tanks
    :type M: int
    :param n: Number of tanks captured
    :type n: int
    :returns: List of captured tank IDs
    :rtype: list
    """
    return sorted(np.random.choice(range(1, M + 1), size=n, replace=False).tolist())

def simulate_estimators(M: int, n: int, num_trials: int) -> dict:
    """
    Simulates the German Tank Problem multiple times and computes
    all three estimators for each.
    
    :param M: Total number of tanks
    :type M: int
    :param n: Number of tanks to capture per sim
    :type n: int
    :param num_trials: Number of simulations
    :type num_trials: int
    :returns: Dictionary with lists of estimates for each estimator
    :rtype: dict
    """
    mle_estimates = []
    mean_estimates = []
    mvu_estimates = []
    
    for _ in range(num_trials):
        captured = tank_capture(M, n)
        mle_estimates.append(mle_M(captured))
        mean_estimates.append(mean_M(captured))
        mvu_estimates.append(mvu_M(captured))
    
    return {
        'mle': mle_estimates,
        'mean': mean_estimates,
        'mvu': mvu_estimates
    }

def run_simulations(M_values: list, num_trials: int) -> pd.DataFrame:
    """
    Runs simulations for all (M, n) combinations,
    computing mean and variance of each estimator.
    
    :param M_values: List of M values to test
    :type M_values: list
    :param num_trials: Number of trials per (M, n) combination
    :type num_trials: int
    :returns: DataFrame with simulation results
    :rtype: pd.DataFrame
    """
    n_percentages = [p * 0.01 for p in range(5, 100, 5)]
    results = []
    
    for M in M_values:
        n_values = [max(1, int(M * p)) for p in n_percentages]
        
        for n in n_values: 
            estimates = simulate_estimators(M, n, num_trials)
            
            results.append({
                'M': M,
                'n': n,
                'mle_mean': np.mean(estimates['mle']),
                'mle_var': np.var(estimates['mle']),
                'mean_mean': np.mean(estimates['mean']),
                'mean_var': np.var(estimates['mean']),
                'mvu_mean': np.mean(estimates['mvu']),
                'mvu_var': np.var(estimates['mvu'])
            })
    
    df = pd.DataFrame(results)
    return df
