import numpy as np
import incompleteCholesky as IC
import LLTSolver as LLT


def PrecCGSolver(A: np.array, b: np.array, delta=1.0e-6, verbose=0):

    if verbose:
        print('Start PrecCGSolver...')

    countIter = 0

    L = IC.incompleteCholesky(A)

    x_k = LLT.LLTSolver(L, b)

    r = A@x_k-b

    d_k = - LLT.LLTSolver(L,r)

    # MISSING CODE
    while np.linalg.norm(r) > delta and countIter < 1000:
        rho_k = np.transpose(d_k)@A@d_k
        t_k = ( np.transpose(r)@LLT.LLTSolver(L,r) ) / rho_k
        x_k = x_k + t_k*d_k
        r_old = r
        r = r_old + t_k*(A@d_k)
        beta_k = ( np.transpose(r) @ LLT.LLTSolver(L,r) ) / (np.transpose(r_old) @ LLT.LLTSolver(L,r_old))
        d_k = - LLT.LLTSolver(L,r) + beta_k*d_k
        countIter = countIter + 1
        if verbose:
            print('STEP ', countIter, ': norm of residual is ', np.linalg.norm(r))

    if verbose:
        print('precCGSolver terminated after ', countIter, ' steps with norm of residual being ', np.linalg.norm(r))
    x = x_k
    return x