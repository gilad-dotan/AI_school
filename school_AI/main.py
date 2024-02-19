import numpy as np
from Layer import Layer
from Network import Network
from scalerFunctions import *
from CostFunctions import *


def main():
    x = np.array([[-1],
                  [1]])

    y = np.array([[0, 1],
                  [1, 0]])

    net = Network([Layer(1, 2, scalerFunction=Sigmoid)])
    print(net.calculate(x))

    print(nnCostFunction(net, x, y, addRegTerm=True))
    print(nnCostFunction(net, x, y, addRegTerm=False))

    print("-" * 20)

    net.setTheta([np.array([[1000, -1000]])])
    print(net.calculate(x))
    print(nnCostFunction(net, x, y, addRegTerm=True))
    print(nnCostFunction(net, x, y, addRegTerm=False))

if __name__ == "__main__":
    main()