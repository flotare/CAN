from typing import Callable
import scipy.linalg as la

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
        self,
        f: Callable[[float], float],
        x0: float,
        tol: float,
        f_prime: Callable[[float], float] = None,
        max_iter: int = 1000,
    ) -> list[float]:
        h = np.sqrt(tol)

        list_x_k = [x0]

        for _ in range(max_iter):
            if abs(f(list_x_k[-1])) < tol:
                return list_x_k
            if f_prime is None:
                f_prime_x_k = (f(list_x_k[-1] + h) - f(list_x_k[-1] - h)) / (2 * h)
            else:
                f_prime_x_k = f_prime(list_x_k[-1])
            if abs(f_prime_x_k) < 1e-14:
                raise ZeroDivisionError(f"Dérivée trop petite en x = {list_x_k[-1]}")
            list_x_k.append(list_x_k[-1] - f(list_x_k[-1]) / f_prime_x_k)
            if abs(list_x_k[-1] - list_x_k[-2]) < tol:
                return list_x_k

        raise RuntimeError(
            f"Méthode de Newton non convergé après {max_iter} itérations"
        )

    def newton_nD(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        tol: float,
        J: Callable[[np.ndarray], np.ndarray] = None,
        max_iter: int = 1000,
    ) -> list[np.ndarray]:
        
        list_x_k = [x0]
        
        if J is None:
            n = x0.shape[0]
            J_k = np.eye(n)
            
        for _ in range(max_iter):
            x_k = list_x_k[-1]
            f_x_k = f(x_k)
            
            if np.linalg.norm(f_x_k) < tol:
                return list_x_k
            
            if J is None and len(list_x_k) > 1:
                delta_f = f_x_k - f(list_x_k[-2])
                delta_x = x_k - list_x_k[-2]
                J_k = J_k + np.outer((delta_f - J_k @ delta_x), delta_x) / (delta_x @ delta_x)
            else:
                J_k = J(x_k) if J is not None else J_k
                
            y_k = la.solve(J_k, f_x_k)
            x_next = x_k - y_k
            list_x_k.append(x_next)
            if np.linalg.norm(x_next - x_k) < tol:
                return list_x_k
            
        raise RuntimeError(
            f"Méthode de Newton non convergé après {max_iter} itérations"
        )
