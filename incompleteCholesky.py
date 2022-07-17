import numpy as np

def incompleteCholesky(A: np.array, alpha=1.0e-3, delta=1.0e-6, verbose=0):
    L = np.copy(A)
    dim = np.shape(L)
    n = dim[0]
    if n != dim[1]:
        raise ValueError('A has wrong dimension.')

    if np.max(np.abs(A - A.T) > 1.0e-6):
        raise ValueError('A is not symmetric.')

    if alpha < 0:
        raise ValueError('range of alpha is wrong!')

    if delta < 0:
        print('Warning: negative delta detected, sparsity is not preserved.')

    if verbose:
        print('Start incompleteCholesky...')

    # MISSING CODE

    for k in range(n):
        # a)
        L[k, k] = np.sqrt(np.max([L[k, k], alpha]))
        # b)
        for i in range(k + 1, n):
            if np.abs(L[i, k]) > delta:
                L[i, k] = L[i, k] / L[k, k]
            else:
                L[i, k] = 0
        # c)
        for j in range(k + 1, n):
            for i in range(j, n):
                if np.abs(L[i, j]) > delta:
                    L[i, j] = L[i, j] - L[i, k] * L[j,k]

        for i in range(n):
            for j in range(i+1,n):
                L[i,j] = 0
        # MISSING CODE
        if verbose:
            residualmatrix = A - L @ L.T
            residual = np.max(np.abs(residualmatrix))
            print('IncompleteCholesky terminated with norm of residual:')
            print(residual)

    return L
