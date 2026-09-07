import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = np.dot(x, w) + b
        y_hat = 1.0 / (1.0 + np.exp(-z))
        # L = 0.5 * (y_hat - y_true)**2
        dl_dw = (y_hat - y_true) * y_hat * (1 - y_hat) * x
        dl_db = (y_hat - y_true) * y_hat * (1 - y_hat)
        return (np.round(dl_dw, 5), np.round(dl_db, 5))
