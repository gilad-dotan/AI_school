import numpy as np
import copy
from scalerFunctions import Sigmoid

class Layer:

    _theta = None
    _numOfNodes = 1
    _scalerFunction = None
    _numOfOutputNodes=1
    _addA0 = True

    def __init__(self, numOfNodes: int, numOfOutputNodes: int, scalerFunction=Sigmoid, theta: np.ndarray=None, addA0=None):
        if addA0 is not None:
            self._addA0 = addA0
        self._numOfOutputNodes = numOfOutputNodes
        self._scalerFunction = scalerFunction
        self._numOfNodes = numOfNodes
        if theta is None:
            if self._addA0:
                self._theta = np.random.rand(1 + self._numOfNodes, self._numOfOutputNodes)
            else:
                self._theta = np.random.rand(self._numOfNodes, self._numOfOutputNodes)
        else:
            self._theta = theta

    def calcOutput(self, x: np.ndarray, addA0=None, applyScaler=True):
        if addA0 is None:
            addA0 = self._addA0

        if addA0:
            # adding a0
            x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
        output = np.dot(x, self._theta)

        if applyScaler:
            return self._scalerFunction.calculate(output)
        return output

    @property
    def scaler(self):
        return self._scalerFunction

    def setTheta(self, newTheta: np.ndarray):
        if newTheta.shape != self._theta.shape:
            raise ValueError("The dimension of the new theta does not match the old shape of theta\n"
                             f"newTheta: {newTheta.shape} != {self._theta.shape}")

        self._theta = copy.deepcopy(newTheta)

    def getTheta(self):
        return self._theta

    def getNumOfNodes(self):
        return self._theta.shape[0]

    def getNumOfOutputNodes(self):
        return self._theta.shape[1]