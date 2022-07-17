import numpy as np


def projectedBacktrackingSearch(f, P, x: np.array, d: np.array, sigma=1.0e-4, verbose=0):
    xp = P.project(x)
    gradx = f.gradient(xp)
    decrease = gradx.T @ d

    if decrease >= 0:
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 1:
        raise TypeError('range of sigma is wrong!')

    if verbose:
        print('Start projectedBacktrackingSearch...')

    beta = 0.5
    t = 1

    def phi(t):
        value = f.objective(P.project(xp+t*d))
        return value

    f0 = phi(0)

    while phi(t) > f0 - sigma/t * np.linalg.norm(xp - P.project(xp - t * gradx))**2:
        t = t * beta

    if verbose:
        print('projectedBacktrackingSearch terminated with t=', t)

    return t
