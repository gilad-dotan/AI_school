import numpy as np
from Layer import Layer
from Network import Network
import pandas as pd
from scalerFunctions import *
from CostFunctions import *
from trainers import *
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("apple_quality.csv")
    df = df.drop(columns=["A_id"])



    y = df["Quality"].to_numpy()
    y[y == 'good'] = 1
    y[y == 'bad'] = 0
    y = y.reshape((y.shape[0], 1)).astype(float)

    df = df.drop(columns=["Quality"])
    x = df.to_numpy().astype(float)

    net = Network([Layer(7, 6, addA0=False, scalerFunction=Sigmoid),
                   Layer(6, 1)])

    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=42)
    #X_val, X_test, y_val, y_test = train_test_split(x, y, test_size=0.5, random_state=42)


    trainer = GradientDecent(net, alpha=0.001)
    trainer.train(X_train, y_train, 100000, nnCostFunction, gama=0.01)
    print(nnCostFunction(net, X_val, y_val, gama=0.01))

if __name__ == "__main__":
    main()
