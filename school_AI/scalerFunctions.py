import numpy as np

class Linear:
    def __init__(self):
        pass

    @staticmethod
    def calculate(x):
        return x

    @staticmethod
    def getGradient(x):
        return 1

class Sigmoid:
    def __init__(self):
        pass

    @staticmethod
    def calculate(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def getGradient(x):
        return Sigmoid.calculate(x) * (1 - Sigmoid.calculate(x));