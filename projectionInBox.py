import numpy as np

class projectionInBox:
  
    def __init__(self, a: np.array, b: np.array, eps=1.0e-6):
        self.a = a
        self.b = b
        self.eps = eps
        if np.min(b - a) < eps:
            raise TypeError('a and b forming box is degenerate.')
        
    def project(self, x: np.array):
        n = x.shape[0]
        projectedX = x.copy()
        for i in range(n):
            if x[i, 0] < self.a[i, 0]:
                projectedX[i, 0] = self.a[i, 0]
          
            if x[i, 0] > self.b[i, 0]:
                projectedX[i, 0] = self.b[i, 0]
                
        return projectedX
    
    def activeIndexSet(self, x: np.array):
        n = x.shape[0]
        myList = []
        for i in range(n):
            if x[i, 0] <= self.a[i, 0]+self.eps or x[i, 0] >= self.b[i, 0]-self.eps:
                myList.append(i)

        return myList
