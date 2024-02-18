import numpy as np
import copy
from scalerFunctions import Sigmoid

class Layer:

    _theta = None
    _numOfNodes = 1
    _scalerFunction = None

    def __init__(self, numOfNodes: int, scalerFunction=Sigmoid, theta: np.ndarray=None):
        self._scalerFunction = scalerFunction
        self._numOfNodes = numOfNodes
        if theta is None:
            self._theta = np.random.rand(self._numOfNodes, 1)
        else:
            self._theta = theta

    def calcOutput(self, x):
        output = np.dot(x, self._theta)

        return self._scalerFunction.calculate(output)

    def setTheta(self, newTheta: np.ndarray):
        if newTheta.shape != self._theta.shape:
            raise ValueError("The dimension of the new theta does not match the old shape of theta")

        self._theta = copy.deepcopy(newTheta)

    def getTheta(self):
        return self._theta