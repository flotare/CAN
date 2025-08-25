from typing import Callable
import numpy as np


class MoindreCarre():
    def __init__(self, list_x_k: np.ndarray, list_y_k: np.ndarray):
        self.list_x_k = np.array(list_x_k, dtype=float).flatten()
        self.list_y_k = np.array(list_y_k, dtype=float).flatten()
        
    def f_degre(self, n: int) -> Callable[[int | float], float]:
        X = np.vander(self.list_x_k, N=n+1, increasing=True)
        y = self.list_y_k.reshape(-1, 1)

        theta = np.linalg.inv(X.T @ X) @ X.T @ y

        def poly(x: np.ndarray) -> np.ndarray:
            x = np.array(x, dtype=float)
            X_eval = np.vander(x, N=n+1, increasing=True)
            return (X_eval @ theta).flatten()

        return poly