from typing import Callable
import sympy as sp

import numpy as np


class ResolveEquation:
    def dichotomie(
        self, f: Callable[[float], float], a: float, b: float, tol: float
    ) -> list[float]:
        if f(a) * f(b) >= 0:
            raise ValueError("l'interval donné doit satisfaire : f(a)*f(b) < 0")

        a_k = a
        b_k = b
        m_k = [(a_k + b_k) / 2]

        while abs(f(m_k[-1])) > tol:
            if f(a_k) * f(m_k[-1]) < 0:
                b_k = m_k[-1]
            else:
                a_k = m_k[-1]
            m_k.append((a_k + b_k) / 2)
        return m_k

    def point_fixe(
        self,
        g: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        tol: float,
        max_iter: int = 1000,
    ) -> list[float]:
        x0 = np.array(x0, dtype=float)
        list_x_k = [x0.copy()]

        for _ in range(max_iter):
            x_next = g(list_x_k[-1])
            if np.linalg.norm(x_next - list_x_k[-1]) < tol:
                list_x_k.append(x_next)
                return list_x_k
            list_x_k.append(x_next)

        raise RuntimeError(f"Point fixe non convergé après {max_iter} itérations")

    def newton(
        self, f: Callable[[float], float], x0: float, tol: float, max_iter: int = 1000
    ) -> list[float]:
        h = np.sqrt(tol)

        list_x_k = [x0]

        for _ in range(max_iter):
            if abs(f(list_x_k[-1])) < tol:
                return list_x_k
            f_prime_x_k = (f(list_x_k[-1] + h) - f(list_x_k[-1] - h)) / (2 * h)
            if abs(f_prime_x_k) < 1e-14:  # éviter division par zéro
                raise ZeroDivisionError(f"Dérivée trop petite en x = {list_x_k[-1]}")
            list_x_k.append(list_x_k[-1] - f(list_x_k[-1]) / f_prime_x_k)
            if abs(list_x_k[-1] - list_x_k[-2]) < tol:
                return list_x_k
            
        raise RuntimeError(
            f"Méthode de Newton non convergé après {max_iter} itérations"
        )
