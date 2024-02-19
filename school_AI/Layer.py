import numpy as np
import copy
from scalerFunctions import Sigmoid

class Layer:

    _theta = None
    _numOfNodes = 1
    _scalerFunction = None
    _numOfOutputNodes=1

    def __init__(self, numOfNodes: int, numOfOutputNodes: int, scalerFunction=Sigmoid, theta: np.ndarray=None):
        self._numOfOutputNodes = numOfOutputNodes
        self._scalerFunction = scalerFunction
        self._numOfNodes = numOfNodes
        if theta is None:
            self._theta = np.random.rand(self._numOfNodes, self._numOfOutputNodes)
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

    def getNumOfNodes(self):
        return self._theta.shape[0]

    def getNumOfOutputNodes(self):
        return self._theta.shape[1]