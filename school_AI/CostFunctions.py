import numpy as np
from Network import Network

def nnCostFunction(net: Network, X: np.ndarray, y: np.ndarray, gama: int = 0.1, addRegTerm=True) -> int:
    """
    this function computes the cost value for the neural network
    :param net: the network
    :param X: the input data
    :param y: the output data
    :param gama: the lambda value
    :param addRegTerm: whether to add the reg term
    :return: (int) the cost value
    """

    m = X.shape[0]

    RegTerm = 0
    Cost = 0

    k = (y == 0)
    Hypothesis = net.calculate(X)

    # calculating by parts
    firstTerm = k * np.log(1-Hypothesis)
    secondTerm = (~k) * np.log(Hypothesis)
    firstTerm[np.isnan(firstTerm)] = 0
    secondTerm[np.isnan(secondTerm)] = 0

    Cost += -(firstTerm + secondTerm)
    
    Cost = Cost.sum()
    Cost /= m

    if addRegTerm:
        thetas = net.getThetas()
        for theta in thetas:
            RegTerm += pow(theta, 2).sum()

        RegTerm = (gama * RegTerm) / (2 * m)

    J = Cost + RegTerm

    return J
    