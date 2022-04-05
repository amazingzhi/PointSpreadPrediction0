from sklearn.ensemble import RandomForestRegressor


class Random_Forest:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def model_train(self):
        clf = RandomForestRegressor(bootstrap=False)
        clf.fit(self.X, self.Y)
        return clf
