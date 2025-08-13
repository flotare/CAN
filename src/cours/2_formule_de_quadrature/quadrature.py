from typing import Callable
import numpy as np


class Quadrature:
    def __init__(
        self,
        f: Callable[[int | float], float],
        a: int = 0,
        b: int | float = 1,
        n_interval: int = 1,
    ):
        self.f = f
        self.a = a
        self.b = b
        self.n_interval = n_interval

    def rectangle_gauche(self) -> float:
        list_x = np.linspace(self.a, self.b, self.n_interval + 1)

        aire = 0.0
        for k in range(self.n_interval):
            a = list_x[k]
            b = list_x[k + 1]
            aire += (b - a) * self.f(a)
        return aire

    def rectangle_droite(self) -> float:
        list_x = np.linspace(self.a, self.b, self.n_interval + 1)

        aire = 0.0
        for k in range(self.n_interval):
            a = list_x[k]
            b = list_x[k + 1]
            aire += (b - a) * self.f(b)
        return aire

    def trapeze(self) -> float:
        aire = (self.rectangle_gauche() + self.rectangle_droite()) / 2
        return aire

    def simpson(self) -> float:
        list_x = np.linspace(self.a, self.b, 2 * self.n_interval + 1)
        
        aire = 0
        for k in range(self.n_interval):
            a, m, b = list_x[2 * k: 2 * k + 3]
            aire += (b - a) * (self.f(a) + 4 * self.f(m) + self.f(b)) / 6
        return aire