import copy

import numpy as np

from Layer import Layer

class Network:

    _numOfLayers = 2
    _layers: list = []

    def __init__(self, layers: list):
        self._layers = layers
        self._numOfLayers = len(layers)

    def calculate(self, x):

        output = copy.deepcopy(x)
        for layer in self._layers:
            output = layer.calcOutput(output)

        return output

    def setTheta(self, Thetas: list):
        for newTheta, layer in zip(Thetas, self._layers):
            layer.setTheta(newTheta)

    def getThetas(self):
        thetas = []
        for layer in self._layers:
            thetas.append(layer.getTheta())

        return thetas

    def getNumOfLabels(self) -> int:
        return self._layers[-1].getNumOfOutputNodes()

    def calculateGradients(self, X, y, gama):

        m = X.shape[0]

        thetas = self.getThetas()
        thetasGrads = []

        a = []
        z = []

        for layer in self._layers:
            X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
            a.append(X)

            X = layer.calcOutput(X, addA0=False, applyScaler=False)
            z.append(X)

            X = layer.scaler.calculate(X)


        Hypothesis = self.calculate(X)

        deltas = []
        deltas.insert(0, Hypothesis - y)

        for i in range(len(thetas), 1, -1):
            print(f"i = {i}")
            delta = np.dot(self._layers[i].getTheta(), deltas[0])
            delta = delta * self._layers[i].scaler.getGradient(z[i])


        for i in range(len(thetas)):
            tempGrad = (np.dot(deltas[i].T, a[i])) / m
            theta = gama * thetas[i]
            theta[0][0] = 0
            tempGrad += theta
            thetasGrads.append(tempGrad)

        return thetasGrads