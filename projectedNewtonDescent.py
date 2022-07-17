import numpy as np
import projectedBacktrackingSearch as PB
import PrecCGSolver as PCG



def projectedNewtonDescent(f, P, x0: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start projectedNewtonDescent...')

    countIter = 0
    xp = P.project(x0)
    B_k = f.hessian(xp)

    while np.linalg.norm(xp - P.project(xp - f.gradient(xp)) ) > eps:
        active_set = P.activeIndexSet(xp) # construct active index set as list

        B_k[active_set,:] = 0 # row set zero all active indices
        B_k[:,active_set] = 0 # column set zero all active indices
        B_k[active_set, active_set] = 1 # set 1 diagonals of active indices

        A = B_k
        b = - f.gradient(xp)
        d_k = PCG.PrecCGSolver(A, b, 1.0e-6, 0)

        if f.gradient(xp).T @ d_k > 0: # check descent direction
            # set steepest direction
            #B_k = 1
            d_k = - f.gradient(xp)

        t_k = PB.projectedBacktrackingSearch(f, P, xp, d_k, 1.0e-4, 0)

        #xk <-- P(xk + tkdk)
        xp = P.project(xp + t_k*d_k)

        B_k = f.hessian(xp) # update B_k

        countIter = countIter + 1


    if verbose:
        print('projectedNewtonDescent terminated after ', countIter, ' steps')

    return xp #Terminated in 15 steps
