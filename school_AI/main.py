import numpy as np
import scalerFunctions

def main():
    print(scalerFunctions.Sigmoid.calculate(np.array([[-4, -3, -2],
                                                      [-1, 0, 1],
                                                      [2, 3, 4]])))

if __name__ == "__main__":
    main()