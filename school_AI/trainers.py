from Network import Network
import numpy as np

class GradientDecent:
    _alpha: int = 0.01
    _net: Network = None

    def __init__(self, net, alpha=None):
        self._net = net

        if alpha is not None:
            self._alpha = alpha

    def train(self, X: np.ndarray, y: np.ndarray, iterations: int, costFunc, gama=0.001):
        for i in range(iterations):
            thetas = self._net.getThetas()
            grads = self._net.calculateGradients(X, y, gama=gama)

            newThetas = []

            for theta, grad in zip(thetas, grads):
                newThetas.append(theta - self._alpha * grad.T)

            self._net.setTheta(newThetas)

            print(f"iter No. {i+1}, J = {costFunc(net=self._net, X=X, y=y, gama=gama, addRegTerm=True)}")

        return self._net