import numpy as np


class augmentedLagrangianObjective:

    def __init__(self, f, h, alpha, gamma):
        if gamma <= 0:
            raise TypeError('range of gamma is wrong!')

        self.f = f
        self.h = h
        self.alpha = alpha
        self.gamma = gamma

    def objective(self, x: np.array):
        # x -> A(x) = f(x) + alpha*h(x)+ 0.5*gamma*h(x)**2
        myObjective = self.f.objective(x) + self.alpha * self.h.objective(x) + 0.5 * self.gamma * (self.h.objective(x))**2

        return myObjective

    def gradient(self, x: np.array):
        # h'(x) (a + b h(x)) + f'(x)
        myGradient = self.h.gradient(x) * self.alpha + self.h.gradient(x) * self.gamma * self.h.objective(x) + self.f.gradient(x)

        return myGradient

    def hessian(self, x: np.array):
        # h''(x) (a + b h(x)) + b h'(x)^2 + f''(x)
        #myHessian = self.alpha * self.h.hessian(x) + self.gamma * self.h.objective(x) * self.h.hessian(x) + self.gamma * (self.h.gradient(x))**2 + self.f.hessian(x)
        #myHessian = self.h.hessian(x) * self.alpha + self.h.hessian(x) * self.gamma * self.h.objective(x) + self.gamma * ( self.h.gradient(x) )**2 + self.f.hessian(x)
        myHessian = self.f.hessian(x) + self.alpha * self.h.hessian(x) + self.gamma * self.h.objective(x) * self.h.hessian(x) + self.gamma * ( self.h.gradient(x) @ (self.h.gradient(x)).T )
        return myHessian
