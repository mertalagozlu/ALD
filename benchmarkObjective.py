import numpy as np


class benchmarkObjective:

    def __init__(self, p: np.array):
        self.p = p

    def objective(self, x: np.array):
        checkargs(x)
        f = self.p[0, 0] * (x[1, 0] + 1) * x[0, 0] ** 2 + np.exp(self.p[1, 0] * x[2, 0] + 1) * x[1, 0] ** 2 \
            + self.p[2, 0] * np.sqrt(x[0, 0] + 1) * x[2, 0] ** 2
        return f

    def gradient(self, x: np.array):
        checkargs(x)
        f_dx0 = 2 * self.p[0, 0] * (x[1, 0] + 1) * x[0, 0] + self.p[2, 0] / 2 * x[2, 0] ** 2 / np.sqrt(x[0, 0] + 1)
        f_dx1 = self.p[0, 0] * x[0, 0] ** 2 + 2 * np.exp(self.p[1, 0] * x[2, 0] + 1) * x[1, 0]
        f_dx2 = self.p[1, 0] * np.exp(self.p[1, 0] * x[2, 0] + 1) * x[1, 0] ** 2 + 2 * self.p[2, 0] * np.sqrt(x[0, 0] + 1) * x[2, 0]
        g = np.array([[f_dx0], [f_dx1], [f_dx2]])
        return g

    def hessian(self, x: np.array):
        checkargs(x)
        f_dx00 = 2 * self.p[0, 0] * (x[1, 0] + 1) - self.p[2, 0] / 4 * x[2, 0] ** 2 / np.sqrt((x[0, 0] + 1) ** 3)
        f_dx01 = 2 * self.p[0, 0] * x[0, 0]
        f_dx02 = self.p[2, 0] * x[2, 0] / np.sqrt(x[0, 0] + 1)
        f_dx11 = 2 * np.exp(self.p[1, 0] * x[2, 0] + 1)
        f_dx12 = 2 * self.p[1, 0] * np.exp(self.p[1, 0] * x[2, 0] + 1) * x[1, 0]
        f_dx22 = self.p[1, 0] ** 2 * np.exp(self.p[1, 0] * x[2, 0] + 1) * x[1, 0] ** 2 + 2 * self.p[2, 0] * np.sqrt(x[0, 0] + 1)
        h = np.array([[f_dx00, f_dx01, f_dx02], [f_dx01, f_dx11, f_dx12], [f_dx02, f_dx12, f_dx22]])
        return h

    def setParameters(self, p: np.array):
        self.p = p

    def parameterGradient(self, x: np.array):
        R_dp1 = (x[1, 0] + 1) * x[0, 0] ** 2
        R_dp2 = x[2, 0] * np.exp(self.p[1, 0] * x[2, 0] + 1) * x[1, 0] ** 2
        R_dp3 = np.sqrt(x[0, 0] + 1) * x[2, 0] ** 2

        myGradP = np.array([[R_dp1], [R_dp2], [R_dp3]], dtype=float)

        return myGradP

    @staticmethod
    def getXData():
        xdata = np.array([[0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 4.0, 0.0, 8.0, 4.0, 4.0, 4.0, 4.0], [-4.0, -4.0, 4.0, 4.0, -4.0, -4.0, 4.0, 4.0, 0.0, 0.0, 0.0, -4.0, 4.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
        return xdata

    @staticmethod
    def getFData():
        fdata = np.array([[22.0, -522.0, 22.0, 1014.0, 337.0, -207.0, 337.0, 1329.0, 48.0, 0.0, 192.0, -101.0, 283.0, 84.0, 84.0]])
        return fdata

def checkargs(x: np.array):
    if x[0] <= -1:
        raise ValueError('x[0] is not allowed to be smaller than -1')
