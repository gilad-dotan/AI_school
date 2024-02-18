import numpy as np

def nnCostFunction(thetas: list, num_labels: int, X: np.ndarray, y: np.ndarray, gama: int):

    """

    :param thetas: the list of theta values for the network
    :param num_labels:
    :param X: the input data
    :param y: the output data
    :return: (J, grad)
    """
    
    m = X.shape[1]
    X = [ones(m, 1), X]
    
    J = 0

    thetasGrads = []

    for theta in thetas:
        thetasGrads.append(np.zeros(theta.shape))
    
    
    # // COST \\
    RegTerm = (gama * (sum(sum((Theta1(:, 2:) .^ 2))) + sum(sum(Theta2(:, 2:) .^ 2)))) / (2 * m)
    Cost = 0
    c = zeros(m, num_labels)
    Hypothesis = sigmoid([ones(m, 1), sigmoid(X * Theta1.T)] * Theta2.T)
    for i = 1:num_labels
        k = (y == i)
        Hypothesis = sigmoid([ones(m, 1), sigmoid(X * Theta1.T)] * Theta2.T)(:, i)
        (-k .* log(Hypothesis) - (1 - k) .* log(1 - Hypothesis))
        c(:, i) = k
        Cost += (-k .* log(Hypothesis) - (1 - k) .* log(1 - Hypothesis))
    
    sum(Cost)
    Cost = sum(Cost) / m
    J = Cost + RegTerm
    
    # calculating the gradient
    a1 = X
    z2 = a1 * Theta1.T
    a2 = [ones(m, 1), sigmoid(z2)]
    z3 = a2 * Theta2.T
    a3 = sigmoid(z3)
    
    Hypothesis = sigmoid([ones(m, 1), sigmoid(X * Theta1.T)] * Theta2.T)
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
    