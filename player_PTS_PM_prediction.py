import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.metrics import max_error, mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from joblib import dump
from sklearn.tree import DecisionTreeRegressor as DTR
import example_web as ew

predictions_dir = '../predictions/predictions_all_test'
models_dir = '../models/models_all_test'
accuracies_dir = '../accuracies/accuracies_all_test'
File_Path_Train = '../data/processing/all_features_selected_from_EDA.csv'
transform = 'standard'  # 'standard', 'minmax'
Model_Name = 'Lasso'
Type = 'feature_selection'  # 'feature_selection', 'PCA'
year_to_be_test = ''
target_cols = ['PTS', 'PLUS_MINUS']
col_to_drop = ['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'TEAM_ID', 'OPPO_ID', 'TEAM_ABBREVIATION', 'TEAM_CITY',
               'PLAYER_ID', 'PLAYER_NAME', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
               'FT_PCT', 'DREB', 'REB', 'AST', 'PTS', 'PLUS_MINUS']
drop_rows = True  #todo: drop rows with your own conditions in example_web.py at file feature_lable_split() function.
mul_models = True
cols_need_matching_pred = ['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'TEAM_ID', 'OPPO_ID', 'TEAM_ABBREVIATION',
                           'TEAM_CITY', 'PLAYER_ID', 'PLAYER_NAME', 'loc', 'PTS', 'PLUS_MINUS', 'START_POSITION_B',
                           'START_POSITION_C', 'START_POSITION_F', 'START_POSITION_G', 'season_Pos', 'season_Pre',
                           'season_Reg']

def get_matching_info_for_pred(data_path, drop_rows, cols_need):
    df = pd.read_csv(data_path)
    df = df[cols_need]
    if drop_rows:
        """write with your own conditions"""
        df = df[df['season_Pre'] == 0]
        df_bench, df_not_bench = df[df['START_POSITION_B'] == 1], df[df['START_POSITION_B'] == 0]
        return df_bench, df_not_bench
    else:
        return df
df_bench_mat_pred, df_not_bench_mat_pred = get_matching_info_for_pred(File_Path_Train, drop_rows, cols_need_matching_pred)
# read data
for target_col in target_cols:
    if year_to_be_test:
        X_train, y_train, X_test, y_test = ew.read_data_select_year_test_MLmodels(train_path=File_Path_Train,
            year_to_be_test=year_to_be_test, target_col=target_col, col_to_drop=col_to_drop, transform=transform,
            data_pre=Type, drop_rows=drop_rows, mul_models=mul_models)
    else:
        if mul_models:
            X_bench, X_not_bench, y_bench, y_not_bench = ew.read_data_select_year_test_MLmodels(train_path=File_Path_Train,
                year_to_be_test=year_to_be_test, target_col=target_col, col_to_drop=col_to_drop, transform=transform,
                data_pre=Type, drop_rows=drop_rows, mul_models=mul_models)
            datasets = {'bench': [X_bench, y_bench], 'not_bench': [X_not_bench, y_not_bench]}
        else:
            X_train, y_train = ew.read_data_select_year_test_MLmodels(train_path=File_Path_Train,
                year_to_be_test=year_to_be_test, target_col=target_col, col_to_drop=col_to_drop, transform=transform,
                data_pre=Type, drop_rows=drop_rows, mul_models=mul_models)

# train models
    for k,v in datasets.items():
        data_name = k
        X_train, y_train = v[0], v[1]
        clf = 0
        score = 'neg_mean_absolute_error'
        if Model_Name == 'SGD':
            SGD = SGDRegressor()
            SGD_params = {
                'penalty': ['l1', 'l2'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                'fit_intercept': [True, False],
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
            }
            clf = GridSearchCV(SGD, SGD_params, scoring=score, n_jobs=-1, verbose=10)
        elif Model_Name == 'LR':
            tuned_parameters = [{'fit_intercept': [True, False], 'positive': [True, False]}]
            clf = GridSearchCV(LinearRegression(n_jobs=-1), tuned_parameters, scoring=score,
                               n_jobs=-1)
        elif Model_Name == 'Ridge':
            Ridge = Ridge()
            Ridge_params = {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                'fit_intercept': [True, False],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
            }
            clf = GridSearchCV(Ridge, Ridge_params, scoring=score, verbose=10, n_jobs=-1)
        elif Model_Name == 'Lasso':
            Lasso = Lasso()
            Lasso_params = {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                'fit_intercept': [True, False],
            }
            clf = GridSearchCV(Lasso, Lasso_params, scoring=score,
                               n_jobs=-1)
        elif Model_Name == 'DTR':
            DTR_params = {'splitter': ['best', 'random'],
                          "max_depth": list(range(30, X_train.shape[1], 30)),
                          'min_samples_split': list(range(2, 39, 6)),
                          'min_samples_leaf': list(range(1, 21, 3)),
                          'min_weight_fraction_leaf': [0.1, 0.3, 0.5],
                          'max_features': ['auto', 'sqrt', 'log2', None],
                          'min_impurity_decrease': [0.1, 0.3, 0.6, 0.9]
                          }
            clf = RandomizedSearchCV(estimator=DTR(), param_distributions=DTR_params,
                                     scoring=score, n_iter=300, cv=5, random_state=1,
                                     verbose=1, n_jobs=-1)
        clf.fit(X_train, y_train)
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
        y_true, y_pred = y_train, clf.predict(X_train)
        print('rooted_mean_squared_error:' + str(mean_squared_error(y_true, y_pred, squared=False)))
        print('max_error:' + str(max_error(y_true, y_pred)))
        print('mean_absolute_error:' + str(mean_absolute_error(y_true, y_pred)))
        print('r2_score:' + str(r2_score(y_true, y_pred)))

        # save train and test errors
        accuracies_to_save = ['best_score_: ', str(clf.best_score_) + '\n', 'best_params_: ', str(clf.best_params_) + '\n',
                              'rooted_mean_squared_error: ' + str(mean_squared_error(y_true, y_pred, squared=False)) + '\n',
                              'max_error: ' + str(max_error(y_true, y_pred)) + '\n', 'mean_absolute_error: ' + str(mean_absolute_error(y_true, y_pred)) + '\n',
                              'r2_score: ' + str(r2_score(y_true, y_pred)) + '\n']
        with open(f'{accuracies_dir}/{Type}_{score}_{Model_Name}_{transform}_{data_name}_{target_col}_accuracy.txt', 'w') as f:
            f.writelines(accuracies_to_save)

        # save models and predictions
        dump(clf, f'{models_dir}/{Type}_{score}_{Model_Name}_{transform}_{data_name}_{target_col}_model.joblib')  # save model
        # clf = load('filename.joblib') load model
        print()
        # y_pred to csv
        if k == 'bench':
            df_bench_mat_pred[f'{target_col}_pred'] = y_pred
        else:
            df_not_bench_mat_pred[f'{target_col}_pred'] = y_pred

resultCSVPath = f'{predictions_dir}/{Type}_{Model_Name}_{transform}_prediction.csv'
df_pred = pd.concat([df_bench_mat_pred, df_not_bench_mat_pred])
df_pred = df_pred.sort_index(ascending=True)
df_pred.to_csv(resultCSVPath, index=False, na_rep=0)