import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

class Regressor():
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.m = 0
        self.b = 0

        self.lr = 0.001
        self.loss = 0

        self.n = X.shape[0]


    def train(self, epochs=10):
        for i in range(epochs):
            self._update()

    def _update(self):
        loss = 0

        m_gradient = 0
        b_gradient = 0

        for i in range(self.n):
            pred = self._predict(self.X[i])
            y = self.y[i]

            m_gradient += self._dm(pred, y, self.X[i])
            b_gradient += self._db(pred, y)
            loss += self._loss(pred, y)

        self.m -= self.lr * m_gradient
        self.b -= self.lr * b_gradient

        print(loss, end='\r')

    def _loss(self, o, y):
        return (1/self.n) * (np.square(o - y))
    def _predict(self, X):
        return self.m * X + self.b
    def _db(self, o, y):
        return -(2/self.n) * (y - o)
    def _dm(self, o, y, X):
        return -(2/self.n) * X * (y - o)

def main():
    data = pd.read_csv('data.csv')
    X = data['MinTemp'].to_numpy().reshape(-1, 1)
    y = data['MaxTemp'].to_numpy().reshape(-1, 1)

    regressor = Regressor(X, y)
    regressor.train()

    pred = list()
    
    for x in X:
        pred.append(regressor._predict(x))

    print("Linear regression finished with slope of {} and y-intercept of {}".format(regressor.m, regressor.b))

    plt.plot(X, y, 'ro')
    plt.plot(X, pred)
    plt.show()

if __name__ == '__main__':
    main()