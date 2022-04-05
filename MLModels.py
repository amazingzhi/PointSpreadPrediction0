import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFR, GradientBoostingRegressor as GBR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from joblib import dump, load
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor as DTR
import lightgbm as lgb

# setting of the input file, preprocessing method and specific model
File_Path_Train = 'data/Original_data/train&validate.csv'  # 'data/PCA_data/PCA_train.csv'
# 'data/feature_selection_data/train&validate_feature_importance.csv' 'data/Original_data/train&validate.csv'
File_Path_Test = 'data/Original_data/test.csv'  # 'data/PCA_data/PCA_test.csv' 'data/Original_data/test.csv'
# 'data/feature_selection_data/test_feature_selection.csv'
Label_Index = 4  # 0, 4
Preprocessing_Type = 'MinMaxScaler'  # 'StandardScaler', 'MinMaxScaler'
Model_Name = 'RF'  # 'LR' 'RF' 'SVM' 'DTR' 'GBR' 'Light'
Type = 'original'  # 'original', 'feature_selection', 'PCA'

# read data
df_train = pd.read_csv(File_Path_Train)
df_test = pd.read_csv(File_Path_Test)
X_train = df_train.iloc[:, Label_Index + 1:]
X_train = X_train.to_numpy()
Y_train = df_train.iloc[:, Label_Index]
Y_train = Y_train.to_numpy()
X_test = df_test.iloc[:, Label_Index + 1:]
X_test = X_test.to_numpy()
Y_test = df_test.iloc[:, Label_Index]
Y_test = Y_test.to_numpy()

# feature preprocessing
if Preprocessing_Type == 'StandardScaler':
    X_train = preprocessing.StandardScaler().fit_transform(X_train)
    X_test = preprocessing.StandardScaler().fit_transform(X_test)
elif Preprocessing_Type == 'MinMaxScaler':
    X_train = preprocessing.MinMaxScaler().fit_transform(X_train)
    X_test = preprocessing.MinMaxScaler().fit_transform(X_test)

# build models
scores = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'explained_variance']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = 0

    # cross validation for hyper-paramters optimal based on each model
    if Model_Name == 'SVM':
        SVR_params = {
            # randomly sample numbers from 4 to 204 estimators
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
            'gamma': [1, 0.1, 0.01, 0.001],
            # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
            'C': [0.1, 1, 10, 100]
        }
        clf = GridSearchCV(SVR(), SVR_params, scoring=score, verbose=1, n_jobs=-1)  # default 5 folds CV
    elif Model_Name == 'DTR':
        DTR_params = {'splitter': ['best', 'random'],
                      "max_depth": list(range(3, 100)),
                      'min_samples_split': list(range(2, 10)),
                      'min_samples_leaf': list(range(1, 10))
                      }
        clf = RandomizedSearchCV(estimator=DTR(), param_distributions=DTR_params,
                                 scoring=score, n_iter=100, cv=5, random_state=1,
                                 verbose=1, n_jobs=-1)
    elif Model_Name == 'RF':
        RFR_params = {'n_estimators': list(range(3, 200, 10)),
                      "max_depth": list(range(3, 100, 10)),
                      'min_samples_split': list(range(2, 100, 10)),
                      'min_samples_leaf': list(range(1, 100, 10)),
                      'max_features': list(range(5, 100, 10)),
                      'min_impurity_decrease': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
        clf = RandomizedSearchCV(RFR(criterion='absolute_error', n_jobs=-1), RFR_params,
                                 scoring=score, n_iter=100, cv=5, random_state=1, verbose=1,
                                 n_jobs=-1)  # default 5 folds CV
    elif Model_Name == 'LR':
        tuned_parameters = [{'fit_intercept': [True, False], 'positive': [True, False], 'normalize': [True, False]}]
        clf = GridSearchCV(LinearRegression(n_jobs=-1), tuned_parameters, scoring=score,
                           n_jobs=-1)  # default 5 folds CV
    elif Model_Name == 'GBR':
        GBR_params = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                      'learning_rate': [0.1, 0.3, 0.6, 0.9],
                      'n_estimators': list(range(3, 200, 10)),
                      'subsample': [0.1, 0.3, 0.6, 0.9],
                      'alpha': [0.1, 0.3, 0.6, 0.9],
                      "max_depth": list(range(3, 100, 10)),
                      'min_samples_split': list(range(2, 100, 10)),
                      'min_samples_leaf': list(range(1, 100, 10)),
                      'max_features': list(range(5, 40, 5)),
                      'min_impurity_decrease': [0.1, 0.3, 0.6, 0.9]}
        clf = RandomizedSearchCV(estimator=GBR(), param_distributions=GBR_params,
                                 scoring=score,
                                 n_iter=100, cv=5, random_state=1,
                                 verbose=1, n_jobs=-1)
    elif Model_Name == 'Light':
        LightGBM = lgb.LGBMRegressor(bagging_freq=1)
        LightGBM_params = {'objective': ['regression', 'regression_l1', 'huber'],
                           'boosting': ['gbdt', 'dart', 'rf'],
                           'feature_fraction': [0.1, 0.3, 0.6, 0.9],
                           'subsample': [0.1, 0.3, 0.6, 0.9],
                           'num_leaves': list(range(2, 100, 10)),
                           'learning_rate': [0.1, 0.3, 0.6, 0.9],
                           "max_depth": list(range(3, 100, 10)),
                           'n_estimators': list(range(3, 100)),
                           'min_data_in_leaf': list(range(1, 100, 10)),
                           }

        clf = RandomizedSearchCV(estimator=LightGBM, param_distributions=LightGBM_params,
                                 scoring=score,
                                 n_iter=100, cv=5, random_state=1,
                                 verbose=1, n_jobs=-1)

    # train model
    clf.fit(X_train, Y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Best model performance report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()

    # get accuracy measures
    y_true, y_pred = Y_test, clf.predict(X_test)
    print('explained_variance_score:' + str(explained_variance_score(y_true, y_pred)))
    print('max_error:' + str(max_error(y_true, y_pred)))
    print('mean_absolute_error:' + str(mean_absolute_error(y_true, y_pred)))
    print('r2_score:' + str(r2_score(y_true, y_pred)))

    # save models and predictions
    if Type == 'Original':
        dump(clf, 'models/' + score + '_' + Model_Name + '_' + Preprocessing_Type + '_model.joblib')  # save model
        # clf = load('filename.joblib') load model
        print()
        # y_pred to csv
        resultCSVPath = f'predictions/{Model_Name}_{score}_{Preprocessing_Type}_prediction.csv'
        y_pred = pd.DataFrame(y_pred)
        y_pred.to_csv(resultCSVPath, index=False, na_rep=0)
    elif Type == 'feature_selection':
        dump(clf, 'models/feature_selection' + score + '_' + Model_Name + '_' + Preprocessing_Type + '_model.joblib')  # save model
        # clf = load('filename.joblib') load model
        print()
        # y_pred to csv
        resultCSVPath = f'predictions/feature_selection_{Model_Name}_{score}_{Preprocessing_Type}_prediction.csv'
        y_pred = pd.DataFrame(y_pred)
        y_pred.to_csv(resultCSVPath, index=False, na_rep=0)
    elif Type == 'PCA':
        dump(clf, 'models/PCA' + score + '_' + Model_Name + '_' + Preprocessing_Type + '_model.joblib')  # save model
        # clf = load('filename.joblib') load model
        print()
        # y_pred to csv
        resultCSVPath = f'predictions/PCA_{Model_Name}_{score}_{Preprocessing_Type}_prediction.csv'
        y_pred = pd.DataFrame(y_pred)
        y_pred.to_csv(resultCSVPath, index=False, na_rep=0)
