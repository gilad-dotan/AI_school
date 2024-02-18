import numpy as np
from scalerFunctions import Sigmoid

class Layer:

    _theta = None
    _numOfNodes = 1
    _scalerFunction = None

    def __init__(self, numOfNodes, scalerFunction=Sigmoid):
        self._scalerFunction = scalerFunction
        self._numOfNodes = numOfNodes
        self._theta = np.random.rand(self._numOfNodes, 1)

    def calcOutput(self, x):
        output = np.dot(x, self._theta)

        return self._scalerFunction(output)