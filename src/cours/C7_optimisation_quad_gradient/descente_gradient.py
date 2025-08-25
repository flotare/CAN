from matplotlib import pyplot as plt
import numpy as np
import scipy.linalg as la

from cours.C6_estimation_de_valeur_propre.approximation import Approximation


class DescenteGradient:
    def __init__(self, A: np.ndarray, b: np.ndarray):
        if not np.allclose(A, A.T):
            raise ValueError("La matrice A doit être symétrique")
        eigvals = np.linalg.eigvals(A)
        if not np.all(eigvals > 0):
            raise ValueError("La matrice A doit être définie positive")
        self.A = A
        self.b = b
        self.list_x_k = []
        self.name_methode = None

    def solve_pas_constant(
        self, tol: float, x_0: np.ndarray = None, alpha_pas: float = None
    ) -> np.ndarray:
        if x_0 is None:
            x_0 = np.zeros_like(self.b, dtype=float)
            x_0[0] = np.mean(self.b)

        if alpha_pas is None:
            lambda_estimator = Approximation(A=self.A)
            lambda_min = lambda_estimator.puissance_inverse(
                mu=0, tolerance=1e-3, max_iter=100
            )[0]
            lambda_max = lambda_estimator.puissance_iteree(n=10)[0]
            alpha_pas = 2 / (lambda_min + lambda_max)

        self.list_x_k = [x_0.copy()]

        while True:
            d_k = self.b - self.A @ self.list_x_k[-1]
            if la.norm(d_k) < tol:
                break
            self.list_x_k.append(self.list_x_k[-1] + alpha_pas * d_k)

        self.name_methode = "méthode du gradient à pas constant"
        return self.list_x_k[-1]

    def solve_pas_optimal(self, tol: float, x_0: np.ndarray = None) -> np.ndarray:
        if x_0 is None:
            x_0 = np.zeros_like(self.b, dtype=float)
            x_0[0] = np.mean(self.b)
        self.list_x_k = [x_0.copy()]

        while True:
            d_k = self.b - self.A @ self.list_x_k[-1]
            alpha_k = (d_k.T @ d_k) / (d_k.T @ self.A @ d_k)

            if la.norm(d_k) < tol:
                break
            self.list_x_k.append(self.list_x_k[-1] + alpha_k * d_k)

        self.name_methode = "méthode du gradient à pas optimal"
        return self.list_x_k[-1]

    def solve_gradient_conjugue(self, tol: float, x_0: np.ndarray = None) -> np.ndarray:
        if x_0 is None:
            x_0 = np.zeros_like(self.b, dtype=float)
            x_0[0] = np.mean(self.b)

        self.list_x_k = [x_0.copy()]
        x_k = x_0

        r_k = self.b - self.A @ x_k
        p_k = r_k.copy()

        while la.norm(r_k) > tol:
            alpha_k = (r_k.T @ r_k) / (p_k.T @ self.A @ p_k)
            x_k = x_k + alpha_k * p_k
            self.list_x_k.append(x_k.copy())

            r_next = r_k - alpha_k * (self.A @ p_k)
            beta_k = (r_next.T @ r_next) / (r_k.T @ r_k)

            p_k = r_next + beta_k * p_k
            r_k = r_next

        self.name_methode = "méthode du gradient conjugué"
        return self.list_x_k[-1]

    def plot_2D_trajectory(self):
        if not self.list_x_k:
            raise RuntimeError("Il faut d'abord exécuter une méthode de résolution avant de tracer.")
        
        x_star = la.solve(self.A, self.b)

        trajectory = np.hstack(self.list_x_k)
        erreur = trajectory - x_star

        r_max = np.max(np.linalg.norm(erreur, axis=0)) + 0.5

        x1_vals = np.linspace(x_star[0, 0] - r_max, x_star[0, 0] + r_max, 200)
        x2_vals = np.linspace(x_star[1, 0] - r_max, x_star[1, 0] + r_max, 200)

        X1, X2 = np.meshgrid(x1_vals, x2_vals)

        F = (
            0.5
            * (self.A[0, 0] * X1**2 + 2 * self.A[0, 1] * X1 * X2 + self.A[1, 1] * X2**2)
            - self.b[0] * X1
            - self.b[1] * X2
        )

        plt.figure(figsize=(6, 6))
        plt.contour(X1, X2, F, levels=20, cmap="coolwarm")
        plt.plot(
            trajectory[0, :],
            trajectory[1, :],
            "o-",
            color="black",
            label="Descente du gradient",
        )
        plt.scatter([x_star[0, 0]], [x_star[1, 0]], color="red", label="Minimum x*")

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(
            f"Convergence de la {self.name_methode}, Nb. itération = {len(self.list_x_k)}"
        )
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()
        
    def plot_convergence_curve(self, log_scale: bool = True):
        if not self.list_x_k:
            raise RuntimeError("Il faut d'abord exécuter une méthode de résolution avant de tracer.")

        def f(x):
            return 0.5 * (x.T @ self.A @ x) - (self.b.T @ x)

        x_star = la.solve(self.A, self.b)
        f_star = f(x_star)

        valeurs = [float(f(x) - f_star) for x in self.list_x_k]

        plt.figure(figsize=(6, 4))
        plt.plot(range(len(valeurs)), valeurs, "o-", color="blue")
        plt.xlabel("Itérations k")
        plt.ylabel(r"$f(x_k) - f(x^*)$")
        plt.title(f"Courbe de convergence ({self.name_methode})")
        if log_scale:
            plt.yscale("log")
        plt.grid(True, which="both", linestyle="--", linewidth=0.7)
        plt.show()

