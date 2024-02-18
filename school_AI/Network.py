import copy

import numpy as np

from Layer import Layer

class Network:

    _numOfLayers = 2
    _layers: list = []

    def __init__(self, layers: list):
        self._layers = layers
        self._numOfLayers = len(layers)

    def claculate(self, x):

        output = copy.deepcopy(x)

        for layer in self._layers:
            output = layer.calcOutput(x)


        return output

    def setTheta(self, Theta: list):
        for newTheta, layer in zip(Theta, self._layers):
            layer.setTheta(newTheta)
