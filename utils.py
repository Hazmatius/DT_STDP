import numpy as np


def get_lag_diff(array, lag):
    lag_diff = array[lag:] - array[:-lag]
    mean_lag_diff = np.mean(lag_diff)
    std_lag_diff = np.std(lag_diff)
    return mean_lag_diff, std_lag_diff