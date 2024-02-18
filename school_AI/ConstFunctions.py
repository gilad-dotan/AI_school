import numpy as np
from Network import Network

def nnCostFunction(net: Network, num_labels: int, X: np.ndarray, y: np.ndarray, gama: int, scalarFunction):

    """

    :param net: the nn
    :param num_labels:
    :param X: the input data
    :param y: the output data
    :return: (J, grad)
    """

    thetas = net.getThetas()

    m = X.shape[1]
    X = [ones(m, 1), X]

    thetasGrads = []
    RegTerm = 0

    for theta in thetas:
        thetasGrads.append(np.zeros(theta.shape))
        RegTerm += theta.sum()
    
    
    # // COST \\
    RegTerm = (gama * RegTerm) / (2 * m)
    Cost = 0
    c = np.zeros(X.shape)
    for i in range(num_labels):
        k = (y == i)
        Hypothesis = net.calculate(X)
        c(:, i) = k
        Cost += -k * np.log(Hypothesis) - (1 - k) * np.log(1 - Hypothesis))
    
    sum(Cost)
    Cost = sum(Cost) / m
    J = Cost + RegTerm
    
    # calculating the gradient
    a1 = X
    z2 = a1 * Theta1.T
    a2 = [ones(m, 1), scalarFunction.calculate(z2)]
    z3 = a2 * Theta2.T
    a3 = scalarFunction.calculate(z3)
    
    Hypothesis = net.calculate(X)
    #c
    
    delta3 = Hypothesis - c
    delta2 = ((delta3 * Theta2) .* [ones(size(z2,1), 1), sigmoidGradient(z2)])(:, 2:)
    
    Theta1_grad = (delta2.T * a1) / m
    Theta2_grad = (delta3.T * a2) / m
    
    Theta1_gradReg = zeros(size(Theta1_grad))
    Theta2_gradReg = zeros(size(Theta2_grad))
    Theta1_gradReg = [zeros(size(Theta1_grad, 1), 1), (gama / m) * Theta1(:, 2:)]
    Theta2_gradReg = [zeros(size(Theta2_grad, 1), 1), (gama / m) * Theta2(:, 2:)]
    
    Theta1_grad += Theta1_gradReg
    Theta2_grad += Theta2_gradReg
    
    [Theta1_grad.T(:)  Theta2_grad.T(:)]
    
    return J, grad
    