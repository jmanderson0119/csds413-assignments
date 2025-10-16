from sim import run_simulations
from visualize_estimators import plot_estimator_means, plot_estimator_variances

def main():
    results = run_simulations(M_values=[100, 1000, 10000], num_trials=1000)
    plot_estimator_means(results, output_dir='../figures')
    plot_estimator_variances(results, output_dir='../figures')

if __name__ == "__main__": main()
