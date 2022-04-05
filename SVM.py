from sklearn.svm import SVR


class SVM:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def model_train(self):
        clf = SVR(kernel='linear')
        clf.fit(self.X, self.Y)
        return clf