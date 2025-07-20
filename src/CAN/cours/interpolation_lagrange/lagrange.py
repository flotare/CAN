from typing import Callable


class Lagrange:
    def __init__(self):
        self.list_polynomes_lagrange = []

    def interpolation_lagrange_locale(
        self, degre: int, list_x: list[int | float], list_y: list[int | float]
    ) -> Callable[[int | float], float]:
        def P(x: int | float) -> float:
            k = 0
            n = len(list_x)
            while k + degre < n and x > list_x[k + degre]:
                k += degre
            if k + degre >= n:
                k = n - degre - 1

            list_x_k = list_x[k: k + degre + 1]
            list_y_k = list_y[k: k + degre + 1]

            P_local = self.interpolation_lagrange_globale(list_x=list_x_k, list_y=list_y_k)
            return P_local(x)
        return P

    def interpolation_lagrange_globale(
        self, list_x: list[int | float], list_y: list[int | float]
    ) -> Callable[[int | float], float]:
        self.set_list_polynomes_lagrange(list_x=list_x)

        def P(x: int | float) -> float:
            p_interpole = 0
            for index in range(len(list_x)):
                p_interpole += list_y[index] * self.list_polynomes_lagrange[index](x)
            return p_interpole

        return P

    def set_list_polynomes_lagrange(self, list_x: list[int | float]) -> None:
        self.list_polynomes_lagrange.clear()
        for k, x_k in enumerate(list_x):
            x_copie = [x_i for i, x_i in enumerate(list_x) if i != k]
            self.set_polynome_lagrange(x_k=x_k, list_x=x_copie)

    def set_polynome_lagrange(
        self, x_k: int | float, list_x: list[int | float]
    ) -> None:
        def f(x: int | float) -> float:
            produit = 1
            for x_i in list_x:
                produit *= (x - x_i) / (x_k - x_i)
            return produit

        self.list_polynomes_lagrange.append(f)
