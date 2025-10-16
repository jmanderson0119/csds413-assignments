import numpy as np

def mle_M(tank_ids: list) -> int:
    """
    Computes the MLE for the total number of tanks.
    
    :param tank_ids: List of observed tank IDs
    :type tank_ids: list
    :returns: Maximum likelihood estimate of total tanks
    :rtype: int
    """
    return max(tank_ids)

def mean_M(tank_ids: list) -> float:
    """
    Computes the MEAN estimator for the total number of tanks.
    
    :param tank_ids: List of observed tank IDs
    :type tank_ids: list
    :returns: MEAN estimate of total tanks
    :rtype: float
    """
    return 2 * np.mean(tank_ids) - 1

def mvu_M(tank_ids: list) -> float:
    """
    Computes the MVU estimator for the total number of tanks.
    
    :param tank_ids: List of observed tank IDs
    :type tank_ids: list
    :returns: MVU estimate of total tanks
    :rtype: float
    """
    n = len(tank_ids)
    return max(tank_ids) * ((n + 1) / n) - 1
