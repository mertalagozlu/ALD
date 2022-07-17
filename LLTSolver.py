import numpy as np


def LLTSolver(L: np.array, r: np.array, verbose=0):

    if verbose:
        print('Start LLTSolver...')

    n = np.size(r)
    s = r.copy()
    for i in range(n):
        for j in range(i):
            s[i, 0] = s[i, 0] - L[i, j] * s[j, 0]

        if L[i, i] == 0:
            raise Exception('Zero diagonal element detected...')

        s[i, 0] = s[i, 0] / L[i, i]

    y = s

    for i in range(n-1, -1, -1):
        for j in range(n-1, i, -1):
            y[i, 0] = y[i, 0] - L[j, i] * y[j, 0]

        y[i, 0] = y[i, 0] / L[i, i]

    if verbose:
        residual = (L@L.T)@y-r
        print('LLTSolver terminated with residual:')
        print(residual)
    return y
