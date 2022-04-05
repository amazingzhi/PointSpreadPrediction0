from sklearn.model_selection import KFold
import RF as RF
import LR as LR
import SVM as SVM
import sklearn.metrics as mtc
from sklearn import preprocessing
import pandas as pd

# read data
df = pd.read_csv('train&validate.csv')
ColumnName = df.columns.values
ColumnName = ColumnName[5:-1]




X = df.iloc[:, 5:-1]
X = X.to_numpy()
X = preprocessing.MinMaxScaler().fit_transform(X)
Y = df.iloc[:, 4]
Y = Y.to_numpy()

# cross-validation
kf = KFold(n_splits=10)
for train, test in kf.split(X, Y):
    X_train_input = X[train]
    Y_train_input = Y[train]
    X_test_input = X[test]
    Y_test_input = Y[test]

    #'''
    # random forest
    print('----------------------------------------------------------------------------')
    model = RF.Random_Forest(X_train_input, Y_train_input)
    clf = model.model_train()
    Y_predict_input = clf.predict(X_test_input)
    print('RF max error: ' + str(mtc.max_error(Y_test_input, Y_predict_input)))
    print('RF mean absolute error: ' + str(mtc.mean_absolute_error(Y_test_input, Y_predict_input)))
    print('RF The coefficient of determination (1 is perfect prediction): ' + str(mtc.r2_score(Y_test_input, Y_predict_input)))
    key_value = {}
    fi = clf.feature_importances_
    for index, i in enumerate(fi):
        key_value[ColumnName[index]] = i

    key_value = sorted(key_value.items(), key=lambda x: abs(x[1]), reverse=True)
    for aLine in key_value:
        print('RF feature importances: ' + aLine[0] + ',' + str(aLine[1]))
    print('----------------------------------------------------------------------------')
    #'''

    '''
    # linear regrssion
    model = LR.Linear_Regression(X_train_input, Y_train_input)
    clf = model.model_train()
    Y_predict_input = clf.predict(X_test_input)
    print('----------------------------------------------------------------------------')
    print('LR max error: ' + str(mtc.max_error(Y_test_input, Y_predict_input)))
    print('LR mean absolute error: ' + str(mtc.mean_absolute_error(Y_test_input, Y_predict_input)))
    print('LR The coefficient of determination (1 is perfect prediction): ' + str(
        mtc.r2_score(Y_test_input, Y_predict_input)))
    key_value = {}
    fi = clf.coef_
    for index, i in enumerate(fi):
        key_value[ColumnName[index]] = i

    key_value = sorted(key_value.items(), key=lambda x: abs(x[1]), reverse=True)
    for aLine in key_value:
        print('LR feature importances: ' + aLine[0] + ',' + str(aLine[1]))
    print('----------------------------------------------------------------------------')
    '''

    '''
    model = SVM.SVM(X_train_input, Y_train_input)
    clf = model.model_train()
    Y_predict_input = clf.predict(X_test_input)
    print('----------------------------------------------------------------------------')
    print('SVM max error: ' + str(mtc.max_error(Y_test_input, Y_predict_input)))
    print('SVM mean absolute error: ' + str(mtc.mean_absolute_error(Y_test_input, Y_predict_input)))
    print('SVM The coefficient of determination (1 is perfect prediction): ' + str(
        mtc.r2_score(Y_test_input, Y_predict_input)))
    key_value = {}
    fi = clf.coef_[0]
    for index, i in enumerate(fi):
        key_value[ColumnName[index]] = i

    key_value = sorted(key_value.items(), key=lambda x: abs(x[1]), reverse=True)
    for aLine in key_value:
        print('SVM feature importances: ' + aLine[0] + ',' + str(aLine[1]))
    print('----------------------------------------------------------------------------')
    '''