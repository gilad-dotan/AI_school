import copy

from Network import Network
import numpy as np
from CostFunctions import *

class GradientDecent:
    _alpha: int = 0.01
    _net: Network = None

    def __init__(self, net, alpha=None):
        self._net = net

        if alpha is not None:
            self._alpha = alpha

    def train(self, X: np.ndarray, y: np.ndarray, iterations: int, costFunc, gama=0.001, printJump=500):

        pastJ = costFunc(net=self._net, X=X, y=y, gama=gama, addRegTerm=True)
        pastThetas = None

        for i in range(iterations):
            thetas = self._net.getThetas()
            pastThetas = copy.deepcopy(thetas)
            grads = self._net.calculateGradients(X, y, gama=gama)

            newThetas = []

            for theta, grad in zip(thetas, grads):
                newThetas.append(theta - self._alpha * grad.T)


            J = costFunc(net=self._net, X=X, y=y, gama=gama, addRegTerm=True)
            if J > pastJ:
                self._net.setTheta(pastThetas)
                self._alpha = self._alpha / 2
                print(f"at iter No. {i}, new alpha is: {self._alpha}")
            pastJ = J

            self._net.setTheta(newThetas)

            if i % printJump == 0:
                print(f"iter No. {i}, J = {pastJ}")
                #getPercentage(self._net.calculate(X), ytemp)

        return self._net