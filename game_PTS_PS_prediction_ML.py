import pandas as pd
from joblib import dump
import example_web as ew
# global variables
predictions_dir = '../predictions/predictions_19_test_game/past_10_year_train'  # todo: past_x_year_train
models_dir = '../models/models_19_test_game/past_10_year_train'  # todo: past_x_year_train
accuracies_dir = '../accuracies/accuracies_19_test_game/past_10_year_train'  # todo: past_x_year_train
File_Path_Train_pts = '../data/final/final_game_features_pts.csv'  # '../data/final/pts_pca.csv'  '../data/final/final_game_features_pts.csv'
File_Path_Train_ps = '../data/final/final_game_features_ps.csv'  # '../data/final/ps_pca.csv'  '../data/final/final_game_features_ps.csv'
transform = 'standard'  # 'standard', 'minmax'
Model_Name = 'LR'
score = 'neg_mean_absolute_error'
Type = 'feature_selection'  # 'feature_selection', 'PCA'
year_to_be_test = '2019'
target_cols = ['PTS_team', 'pointspread']
col_to_drop = ['GAME_ID', 'GAME_DATE_EST', 'team_id', 'oppo_id', 'PTS_team', 'pointspread', 'FG_PCT_team',
               'FT_PCT_team',
               'FG3_PCT_team', 'AST_team', 'REB_team', 'FGM_team', 'FGA_team', 'FG3M_team', 'FG3A_team', 'FTM_team',
               'FTA_team', 'OREB_team', 'DREB_team', 'STL_team', 'BLK_team', 'TO_team', 'PF_team', 'PLUS_MINUS_team',
                                                                                                   'EFG%_team',
               'PPS_team', 'FIC_team', 'PTS_oppo', 'FG_PCT_oppo', 'FT_PCT_oppo', 'FG3_PCT_oppo',
               'AST_oppo', 'REB_oppo', 'FGM_oppo', 'FGA_oppo', 'FG3M_oppo', 'FG3A_oppo', 'FTM_oppo', 'FTA_oppo',
               'OREB_oppo', 'DREB_oppo', 'STL_oppo', 'BLK_oppo', 'TO_oppo', 'PF_oppo', 'PLUS_MINUS_oppo',
               'EFG%_oppo', 'PPS_oppo', 'FIC_oppo']
cols_need_matching_pred = ['GAME_ID', 'GAME_DATE_EST', 'team_id', 'oppo_id', 'PTS_team', 'pointspread', 'loc']
df = pd.read_csv(File_Path_Train_ps)
df_mat_pred = df[cols_need_matching_pred]
df_mat_pred = df_mat_pred[df_mat_pred['GAME_DATE_EST'].str.contains(year_to_be_test)]
drop_rows = True  # todo: drop rows with your own conditions in example_web.py at file feature_label_split() function.

def add_oppo_pred(row, df):
    game_id = row['GAME_ID']
    oppo_id = row['oppo_id']
    found_line = df[(df['GAME_ID'] == game_id) & (df['team_id'] == oppo_id)]
    oppo_pred_pts = found_line['PTS_team_pred']
    return list(oppo_pred_pts)[0]
# read data
for target_col in target_cols:
    if target_col == 'PTS_team':
        X_train, y_train, X_test, y_test = ew.read_data_select_year_test_MLmodels(train_path=File_Path_Train_pts,
                                                                                      year_to_be_test=year_to_be_test,
                                                                                      target_col=target_col,
                                                                                      col_to_drop=col_to_drop,
                                                                                      transform=transform,
                                                                                      data_pre=Type,
                                                                                      drop_rows=drop_rows,
                                                                                      mul_models=False)
        # build models
        clf = 0
        clf = ew.ML_model_selection(score=score, model_name=Model_Name)
        # train models
        clf.fit(X_train, y_train)
        print(f"Best parameters set found on development set for {target_col}:")
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
        y_true, y_pred = y_test, clf.predict(X_test)
        ew.print_accuracy_measures(y_true, y_pred)

        # save train and test errors
        accuracies_to_save = ew.accuracy_to_save(y_true, y_pred, clf=clf)
        with open(f'{accuracies_dir}/{Type}_{score}_{Model_Name}_{transform}_{target_col}_accuracy.txt',
                  'w') as f:
            f.writelines(accuracies_to_save)

        # save models and predictions
        dump(clf,
             f'{models_dir}/{Type}_{score}_{Model_Name}_{transform}_{target_col}_model.joblib')  # save model
        # clf = load('filename.joblib') load model
        print()
        # calculate predicted pointspread by predicted pts
        df_mat_pred[f'{target_col}_pred'] = y_pred
        df_mat_pred['PTS_oppo_pred'] = df_mat_pred.apply(lambda row: add_oppo_pred(row, df_mat_pred), axis=1)
        df_mat_pred['pointspread_pred_from_pts'] = df_mat_pred['PTS_team_pred'] - df_mat_pred['PTS_oppo_pred']
        # get accuracy measures
        ew.print_accuracy_measures(df_mat_pred['pointspread'], df_mat_pred['pointspread_pred_from_pts'])
        # save train and test errors
        accuracies_to_save = ew.accuracy_to_save(df_mat_pred['pointspread'], df_mat_pred['pointspread_pred_from_pts'], clf=False)
        with open(f'{accuracies_dir}/{Type}_{score}_{Model_Name}_{transform}_{target_col}_to_pointspread_accuracy.txt',
                  'w') as f:
            f.writelines(accuracies_to_save)
    elif target_col == 'pointspread':
        datasets = ew.read_data_select_year_test_MLmodels(train_path=File_Path_Train_ps,
                                                                                  year_to_be_test=year_to_be_test,
                                                                                  target_col=target_col,
                                                                                  col_to_drop=col_to_drop,
                                                                                  transform=transform,
                                                                                  data_pre=Type,
                                                                                  drop_rows=drop_rows,
                                                                                  mul_models=True)
        for k, v in datasets.items():
            data_name = k
            X_train, y_train, X_test, y_test = v[0], v[1], v[2], v[3]
            # build models
            clf = 0
            clf = ew.ML_model_selection(score=score, model_name=Model_Name)
            clf.fit(X_train, y_train)
            print(f"Best parameters set found on development set for {target_col}_{data_name}:")
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
            y_true, y_pred = y_test, clf.predict(X_test)
            ew.print_accuracy_measures(y_true, y_pred)

            # save train and test errors
            accuracies_to_save = ew.accuracy_to_save(y_true, y_pred, clf=clf)
            with open(f'{accuracies_dir}/{Type}_{score}_{Model_Name}_{transform}_{data_name}_{target_col}_accuracy.txt',
                      'w') as f:
                f.writelines(accuracies_to_save)

            # save models and predictions
            dump(clf,
                 f'{models_dir}/{Type}_{score}_{Model_Name}_{transform}_{data_name}_{target_col}_model.joblib')  # save model
            # clf = load('filename.joblib') load model
            print()

            if k == 'home':
                df_home_mat_pred = df_mat_pred[df_mat_pred['loc'] == 1]
                df_home_mat_pred[f'{target_col}_pred'] = y_pred
            else:
                df_away_mat_pred = df_mat_pred[df_mat_pred['loc'] == 0]
                df_away_mat_pred[f'{target_col}_pred'] = y_pred
# gather predictions together
df_pred = pd.concat([df_home_mat_pred, df_away_mat_pred])
df_pred = df_pred.sort_index(ascending=True)
# calculate point spread accuracy including both home and away
ew.print_accuracy_measures(df_pred['pointspread'], df_pred['pointspread_pred'])
# save pointspread accuracies for home and away
accuracies_to_save = ew.accuracy_to_save(df_pred['pointspread'], df_pred['pointspread_pred'], clf=False)
with open(f'{accuracies_dir}/{Type}_{score}_{Model_Name}_{transform}_{target_col}_accuracy.txt',
          'w') as f:
    f.writelines(accuracies_to_save)
# y_pred to csv
resultCSVPath = f'{predictions_dir}/{Type}_{Model_Name}_{transform}_prediction.csv'
df_pred.to_csv(resultCSVPath, index=False, na_rep=0)