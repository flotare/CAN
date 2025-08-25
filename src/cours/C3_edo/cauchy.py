from typing import Callable

import numpy as np


class Cauchy:
    def __init__(
        self,
        f: Callable[[float, float], float],
        t_range: tuple[float, float],
        y0: float,
    ):
        self.t_range = t_range
        self.y0 = y0
        self.f = f

        self.list_t = None
        self.list_y = None

        self.erreur = None

        self.ordre = None

    def euler_explicite(self, n: int) -> None:
        t0, tn = self.t_range
        h = (tn - t0) / n

        self.list_t = np.linspace(t0, tn, n + 1).tolist()
        self.list_y = [self.y0]

        y_k = self.y0

        for t_k in self.list_t[:-1]:
            y_k += h * self.f(t_k, y_k)

            self.list_y.append(y_k)

    def runge_kutta2(self, n: int, alpha: float) -> None:
        t0, tn = self.t_range
        h = (tn - t0) / n

        self.list_t = np.linspace(t0, tn, n + 1).tolist()
        self.list_y = [self.y0]

        y_k = self.y0

        for t_k in self.list_t[:-1]:
            k1 = self.f(t_k, y_k)
            k2 = self.f(t_k + alpha * h, y_k + alpha * h * k1)
            y_k += h * ((1 - 1 / (2 * alpha)) * k1 + (1 / (2 * alpha)) * k2)
            self.list_y.append(y_k)

    def runge_kutta4(self, n: int) -> None:
        t0, tn = self.t_range
        h = (tn - t0) / n

        self.list_t = np.linspace(t0, tn, n + 1).tolist()
        self.list_y = [self.y0]

        y_k = self.y0

        for t_k in self.list_t[:-1]:
            k1 = self.f(t_k, y_k)
            k2 = self.f(t_k + h / 2, y_k + h / 2 * k1)
            k3 = self.f(t_k + h / 2, y_k + h / 2 * k2)
            k4 = self.f(t_k + h, y_k + h * k3)
            y_k += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.list_y.append(y_k)

    def erreur_calc(self, y: Callable[[float], float]):
        self.erreur = []

        for k in range(len(self.list_t)):
            t_k = self.list_t[k]
            y_k = self.list_y[k]

            e_k = y_k - y(t_k)

            self.erreur.append(e_k)

    def estimation_ordre(
        self,
        y: Callable[[float], float],
        methode: Callable[..., None],
        alpha_kutta: float = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        list_n = [int(10**k) for k in np.arange(1, 4, 0.5)]
        list_log_erreur = []
        list_log_n = []

        for n in list_n:
            list_log_n.append(np.log10(n))

            if alpha_kutta is not None:
                methode(n=n, alpha=alpha_kutta)
            else:
                methode(n=n)

            self.erreur_calc(y=y)
            err_max = max(abs(e) for e in self.erreur)
            list_log_erreur.append(np.log10(1 / err_max))

        log_n = np.array(list_log_n)
        log_err = np.array(list_log_erreur)

        x_mean = log_n.mean()
        y_mean = log_err.mean()
        self.ordre = np.sum((log_n - x_mean) * (log_err - y_mean)) / np.sum(
            (log_n - x_mean) ** 2
        )

        return log_n, log_err
