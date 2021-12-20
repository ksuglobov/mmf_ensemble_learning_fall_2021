import numpy as np


def RMSE(y, y_pred):
    """
    RMSE
    """
    return np.sqrt(((y - y_pred) ** 2).mean(axis=-1))
