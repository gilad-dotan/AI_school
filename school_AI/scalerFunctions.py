import numpy as np

class Sigmoid:
    def __init__(self):
        pass

    @staticmethod
    def calculate(x):
        return 1 / (1 + np.exp(-x))