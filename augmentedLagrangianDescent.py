import numpy as np
import projectedNewtonDescent as PD
import augmentedLagrangianObjective as AO

def augmentedLagrangianDescent(f, P, h, x0: np.array, alpha0=0, eps=1.0e-3, delta=1.0e-6, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if delta <= 0 or delta >= eps:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start augmentedLagrangianDescent...')

    countIter = 0
    xp = P.project(x0)
    alpha_k = alpha0
    gamma_k = 10
    eps_k = 1/gamma_k
    delta_k = 1/gamma_k**0.1
    myAugLag = AO.augmentedLagrangianObjective(f, h, alpha_k, gamma_k)


    while np.linalg.norm(xp - P.project(xp - myAugLag.gradient(xp)) ) > eps or np.linalg.norm(h.objective(xp)) > delta:
        xp = PD.projectedNewtonDescent(myAugLag, P, x0, eps_k)

        if np.linalg.norm(h.objective(xp)) <= delta_k:

            alpha_k += gamma_k * h.objective(xp)
            eps_k = np.max([eps_k/gamma_k,eps])
            delta_k = np.max([delta_k/gamma_k**0.9,delta])

        else:

            gamma_k = np.max([10, np.sqrt(gamma_k)]) * gamma_k
            eps_k = 1/gamma_k
            delta_k = 1/gamma_k**0.1

        myAugLag = AO.augmentedLagrangianObjective(f, h, alpha_k, gamma_k)

        countIter = countIter + 1

    if verbose:
        print('augmentedLagrangianDescent terminated after ', countIter, ' steps')
    alphak = alpha_k
    return [xp, alphak]
