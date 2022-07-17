import numpy as np

class quadraticObjective:

    def __init__(self, A: np.array, b: np.array, c: float):
        self.A = A
        self.b = b
        self.c = c

    def objective(self, x: np.array):
        f = 0.5 * (x.T @ (self.A @ x)) + self.b.T @ x + self.c
        return f

    def gradient(self, x: np.array):
        g = self.A @ x + self.b
        return g

    def hessian(self, x: np.array):
        h = self.A
        return h
