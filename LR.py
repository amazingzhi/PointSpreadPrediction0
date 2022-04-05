from sklearn.linear_model import LinearRegression


class Linear_Regression:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def model_train(self):
        clf = LinearRegression()
        clf.fit(self.X, self.Y)
        return clf
