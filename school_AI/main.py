import numpy as np
from Layer import Layer
from Network import Network
from scalerFunctions import *


def main():
    x = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    y = np.array([[3],
                  [12],
                  [21]])

    net = Network([Layer(3, scalerFunction=Linear)])
    print(net.calculate(x))

    net.setTheta([np.array([[8], [-10], [5]])])
    print(net.calculate(x))

if __name__ == "__main__":
    main()