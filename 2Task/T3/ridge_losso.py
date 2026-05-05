from typing import List
import numpy as np


def ridge(e: np.ndarray , b: np.ndarray, lambd):
    return e.T@e + lambd*b.T@b


def lasso(e: np.ndarray , b: np.ndarray, lambd):
    return e.T@e + lambd*np.sum(abs(b))