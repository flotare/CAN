import numpy as np
import scipy.linalg as la


class Approximation:
    def __init__(self, A: np.ndarray):
        self.A = A

    def puissance_iteree(self, n: int) -> tuple[float, np.ndarray]:
        u = np.random.rand(self.A.shape[0])
        u /= np.linalg.norm(u)
        for _ in range(n):
            v = self.A @ u
            u = v / la.norm(v)
        valeur_propre = u @ self.A @ u
        print(
            f"l'estimation de lambda à l'ordre {n} est lambda={valeur_propre:.6f} pour le vecteur propre : {u}"
        )
        return valeur_propre, u

    def puissance_inverse(
        self, mu: float, tolerance: float, max_iter: int = 100
    ) -> tuple[float, np.ndarray]:
        u = np.random.rand(self.A.shape[0])
        u /= np.linalg.norm(u)
        B = self.A - mu * np.eye(self.A.shape[0])
        for _ in range(max_iter):
            y = la.solve(B, u)
            y /= np.linalg.norm(y)
            
            mu = y @ self.A @ y
            B = self.A - mu * np.eye(self.A.shape[0])

            valeur_propre = y @ self.A @ y

            erreur = np.linalg.norm(self.A @ y - valeur_propre * y)

            if erreur < tolerance:
                return valeur_propre, y

            u = y
        raise RuntimeError(
            "Tolérance non atteinte dans le nombre d'itérations maximum."
        )

    def qr_factorisation(self, n: int) -> list[float]:
        A_k = self.A.copy()
        for _ in range(n):
            Q, R = la.qr(A_k)
            A_k = R @ Q
        vals_propres = np.diag(A_k)
        return vals_propres
    
    def qr_factorisation_shift(self, n: int) -> list[float]:
        A_k = self.A.copy()
        for _ in range(n):
            # Choix simple du shift : élément en bas à droite
            mu = A_k[-1, -1]
            # QR sur la matrice décalée
            Q, R = la.qr(A_k - mu * np.eye(A_k.shape[0]))
            # Reconstruction avec shift
            A_k = R @ Q + mu * np.eye(A_k.shape[0])
        vals_propres = np.diag(A_k)
        return vals_propres
