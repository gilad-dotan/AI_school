import numpy as np
from Layer import Layer
from Network import Network
import pandas as pd
from scalerFunctions import *
from CostFunctions import *
from trainers import *
from sklearn.model_selection import train_test_split

def trainApples():
    df = pd.read_csv(r"datasets/apples/apple_quality.csv")
    df = df.drop(columns=["A_id"])

    y = df["Quality"].to_numpy()
    y[y == 'good'] = 1
    y[y == 'bad'] = 0
    y = y.reshape((y.shape[0], 1)).astype(float)

    df = df.drop(columns=["Quality"])
    x = df.to_numpy().astype(float)

    net = Network([Layer(7, 10, addA0=False, scalerFunction=Sigmoid),
                   Layer(10, 10),
                   Layer(10, 1)])

    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

    trainer = GradientDecent(net, alpha=1)
    trainer.train(X_train, y_train, 10000, nnCostFunction, gama=0.05)
    print(nnCostFunction(net, X_val, y_val, gama=0))

    finalValues = net.calculate(X_val)

    for i in range(finalValues.shape[0]):
        if finalValues[i, 0] >= 0.5:
            finalValues[i, 0] = 1
        else:
            finalValues[i, 0] = 0
    print(abs(y_val - finalValues))
    print(1 - abs(y_val - finalValues).sum() / y_val.shape[0], y_val.shape)




def main():
    trainApples()




if __name__ == "__main__":
    main()


def trainObesity():
    def getData():
        df = pd.read_csv(r"datasets/train.csv")
        df = df.drop(columns=["id"])

        X = df["Gender"].to_numpy().reshape((-1, 1))
        X[X == 'Male'] = 1
        X[X == 'Female'] = 0

        X = np.append(X, df["Age"].to_numpy().reshape((-1, 1)), axis=1)
        X = np.append(X, df["Height"].to_numpy().reshape((-1, 1)), axis=1)

        temp = df["family_history_with_overweight"].to_numpy().reshape((-1, 1))
        temp[temp == 'yes'] = 1
        temp[temp == 'no'] = 0
        X = np.append(X, temp, axis=1)

        temp = df["FAVC"].to_numpy().reshape((-1, 1))
        temp[temp == 'yes'] = 1
        temp[temp == 'no'] = 0
        X = np.append(X, temp, axis=1)

        X = np.append(X, df["FCVC"].to_numpy().reshape((-1, 1)), axis=1)
        X = np.append(X, df["NCP"].to_numpy().reshape((-1, 1)), axis=1)

        temp = df["CAEC"].to_numpy().reshape((-1, 1))
        t = np.zeros((temp.shape[0], 1))
        for i in range(len(np.unique(temp))):
            t = np.append(t, temp == np.unique(temp)[i], axis=1)
        t = t[:, 1:]
        X = np.append(X, t, axis=1)

        temp = df["SMOKE"].to_numpy().reshape((-1, 1))
        temp[temp == 'yes'] = 1
        temp[temp == 'no'] = 0
        X = np.append(X, temp, axis=1)

        X = np.append(X, df["CH2O"].to_numpy().reshape((-1, 1)), axis=1)

        temp = df["SCC"].to_numpy().reshape((-1, 1))
        temp[temp == 'yes'] = 1
        temp[temp == 'no'] = 0
        X = np.append(X, temp, axis=1)

        X = np.append(X, df["FAF"].to_numpy().reshape((-1, 1)), axis=1)
        X = np.append(X, df["TUE"].to_numpy().reshape((-1, 1)), axis=1)

        temp = df["CALC"].to_numpy().reshape((-1, 1))
        t = np.zeros((temp.shape[0], 1))
        for i in range(len(np.unique(temp))):
            t = np.append(t, temp == np.unique(temp)[i], axis=1)
        t = t[:, 1:]
        X = np.append(X, t, axis=1)

        temp = df["MTRANS"].to_numpy().reshape((-1, 1))
        t = np.zeros((temp.shape[0], 1))
        for i in range(len(np.unique(temp))):
            t = np.append(t, temp == np.unique(temp)[i], axis=1)
        t = t[:, 1:]
        X = np.append(X, t, axis=1)

        temp = df["NObeyesdad"].to_numpy().reshape((-1, 1))
        labelsList = list(np.unique(temp))
        t = np.zeros((temp.shape[0], 1))
        for i in range(len(np.unique(temp))):
            t = np.append(t, temp == np.unique(temp)[i], axis=1)
        y = t[:, 1:]

        for i in range(temp.shape[0]):
            temp[i, 0] = labelsList.index(temp[i, 0])

        return X.astype(float), y.astype(float), temp.astype(float), labelsList


    X, y, yTemp, labelsList = getData()
    y = np.append(y, yTemp, axis=1)
    print(labelsList)
    print(y)

    net = Network([Layer(24, 24, addA0=False, scalerFunction=Sigmoid),
                   Layer(24, 24),
                   Layer(24, 24),
                   Layer(24, 7)])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    yTemp_train = y_train[:, -1].reshape((-1, 1))
    y_train = y_train[:, :-1]

    yTemp_val = y_val[:, -1].reshape((-1, 1))
    y_val = y_val[:, :-1]

    # yTemp_test = y_test[:, -1].reshape((-1, 1))
    # y_test = y_test[:, :-1]

    print(X_train.shape, y_train.shape)

    trainer = GradientDecent(net, alpha=1)
    net = trainer.train(X_train, y_train, yTemp_train, 1000000, nnCostFunction, gama=0.05, printJump=300)
    print(nnCostFunction(net, X_val, y_val, gama=0))

    thetas = net.getThetas()
    for i in range(len(thetas)):
        np.savetxt(f'tes{i}1.txt', thetas[i], fmt='%d')
        # b = np.loadtxt('test1.txt', dtype=int)
