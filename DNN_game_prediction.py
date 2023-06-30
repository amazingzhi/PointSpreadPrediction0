# import libraries
import torch
import pandas as pd
from torch.utils.data import DataLoader
import example_web as ew
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

""" 
ToDo YOU HAVE TO CREATE 'checkpoint' and 'models' and 'predictions' folders in your working directory
Todo: after you change datasets, put "tensorboard --logdir runs_feature_selected ; runs_PCA" into terminal and
    change working directory. after you have run the program, put "tensorboard --logdir runs_original ; runs_feature_selected ; runs_PCA" into terminal again to see your results in tensorboard. 
todo : change variables with comments after codes when you change datasets"""
# global parameters set up
    # training parameters
n_epochs = 88
k = 3
shuffle = True
random_state = 42
splits = KFold(n_splits=k,shuffle=shuffle,random_state=random_state)
    # model parameters
learning_rates=[1e-4]  #1e-3, 1e-2
batch_sizes=[16]  # , 32, 64
weight_decays=[1e-4]  # , 1e-6
nodes_propotions=[1]  # , 2, 3
denominator_of_input=len(nodes_propotions)
num_layers = ['model_params2', 'model_params3']  # 'model_params1', 'model_params2', 'model_params3',
    # tensorboard
tb = SummaryWriter()
    #read data parameters
train_and_test = True  # True
data_pre = 'feature_selected'  # 'feature_selected', 'PCA'
train_path_ps = '../data/final/final_game_features_ps.csv'  # '../data/final/ps_pca.csv'
predictions_dir = '../predictions/predictions_19_test_game'
models_dir = '../models/models_19_test_game/DNN'
accuracies_dir = '../accuracies/accuracies_19_test_game'
runs_dir = '../models/models_19_test_game/DNN/runs'
checkpoint_dir = '../models/models_19_test_game/DNN/checkpoint'
year_to_be_test = '2019'  # the year to be test dataset.
target_col = 'pointspread'
col_to_drop = ['GAME_ID', 'GAME_DATE_EST', 'team_id', 'oppo_id', 'PTS_team', 'pointspread', 'FG_PCT_team',
               'FT_PCT_team',
               'FG3_PCT_team', 'AST_team', 'REB_team', 'FGM_team', 'FGA_team', 'FG3M_team', 'FG3A_team', 'FTM_team',
               'FTA_team', 'OREB_team', 'DREB_team', 'STL_team', 'BLK_team', 'TO_team', 'PF_team', 'PLUS_MINUS_team',
                                                                                                   'EFG%_team',
               'PPS_team', 'FIC_team', 'PTS_oppo', 'FG_PCT_oppo', 'FT_PCT_oppo', 'FG3_PCT_oppo',
               'AST_oppo', 'REB_oppo', 'FGM_oppo', 'FGA_oppo', 'FG3M_oppo', 'FG3A_oppo', 'FTM_oppo', 'FTA_oppo',
               'OREB_oppo', 'DREB_oppo', 'STL_oppo', 'BLK_oppo', 'TO_oppo', 'PF_oppo', 'PLUS_MINUS_oppo',
               'EFG%_oppo', 'PPS_oppo', 'FIC_oppo']
transform = 'standard'  # 'minmax', 'standard', "maxabs", "robust".
scaler_X = ew.get_scaler(transform)
scaler_y = ew.get_scaler(transform)
resultCSVPath = f'{predictions_dir}/{data_pre}_DNN_{transform}.csv'  # predictions/{data_pre}_DNN_{transform}.csv
cols_need_matching_pred = ['GAME_ID', 'GAME_DATE_EST', 'team_id', 'oppo_id', 'PTS_team', 'pointspread', 'loc']
df = pd.read_csv(train_path_ps)
df_mat_pred = df[cols_need_matching_pred]
df_mat_pred = df_mat_pred[df_mat_pred['GAME_DATE_EST'].str.contains(year_to_be_test)]
drop_rows = True  # todo: drop rows with your own conditions in example_web.py at file feature_label_split() function.
neural_nets = True
## see if cuda available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

def add_oppo_pred(row, df):
    game_id = row['GAME_ID']
    oppo_id = row['oppo_id']
    found_line = df[(df['GAME_ID'] == game_id) & (df['team_id'] == oppo_id)]
    oppo_pred_pts = found_line['PTS_team_pred']
    return list(oppo_pred_pts)[0]

def main():
    datasets = ew.read_data_select_year_test_MLmodels(train_path=train_path_ps,
                                                                              year_to_be_test=year_to_be_test,
                                                                              target_col=target_col,
                                                                              col_to_drop=col_to_drop,
                                                                              transform=transform,
                                                                              data_pre=data_pre,
                                                                              drop_rows=drop_rows,
                                                                              mul_models=True,
                                                                              neural_nets=neural_nets)
    for k,v in datasets.items():
        data_name = k
        train, test = v[0], v[1]


        # training parameters
        param_values = ew.training_parameters(learning_rates=learning_rates,batch_sizes=batch_sizes,
                                              weight_decays=weight_decays,num_layers=num_layers,nodes_propotions=nodes_propotions)


        # training with cross validation
        best_opt, best_model, best_optimizer, best_model_path = ew.cross_validation(param_values=param_values, denominator_of_input=denominator_of_input,
                                                                                    splits=splits, train_data=train, n_epochs=n_epochs,
                                                                                    folds=k, data_pre=data_pre,
                                                                                    transform=transform, runs_dir=runs_dir,
                                                                                    model_dir=models_dir, checkpoint_dir=checkpoint_dir)


        # load test dataset
        if train_and_test:
            test_loader = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
        else:
            test_loader = DataLoader(train, batch_size=1, shuffle=False, drop_last=True)

        # use best model to predict using test dataset.
        predictions, values = best_opt.evaluate(
            test_loader,
            model_ori=best_model,
            optimizer_ori=best_optimizer,
            best_model_path=best_model_path,
            batch_size=1,
            n_features=train.tensors[0].shape[1]
        )
        # predictions = scaler_y.inverse_transform(predictions)
        # values = scaler_y.inverse_transform(values)
        ew.print_accuracy_measures(values, predictions)
        accuracies_to_save = ew.accuracy_to_save(values, predictions, clf=False)
        with open(f'{accuracies_dir}/{data_pre}_DNN_{transform}_{data_name}_{target_col}_accuracy.txt',
                  'w') as f:
            f.writelines(accuracies_to_save)
        if k == 'home':
            df_home_mat_pred = df_mat_pred[df_mat_pred['loc'] == 1]
            df_home_mat_pred[f'{target_col}_pred'] = predictions
        else:
            df_away_mat_pred = df_mat_pred[df_mat_pred['loc'] == 0]
            df_away_mat_pred[f'{target_col}_pred'] = predictions

    # gather predictions together
    df_pred = pd.concat([df_home_mat_pred, df_away_mat_pred])
    df_pred = df_pred.sort_index(ascending=True)
    # calculate point spread accuracy including both home and away
    ew.print_accuracy_measures(df_pred['pointspread'], df_pred['pointspread_pred'])
    # save pointspread accuracies for home and away
    accuracies_to_save = ew.accuracy_to_save(df_pred['pointspread'], df_pred['pointspread_pred'], clf=False)
    with open(f'{accuracies_dir}/{data_pre}_DNN_{transform}_{target_col}_accuracy.txt',
              'w') as f:
        f.writelines(accuracies_to_save)
    # y_pred to csv
    resultCSVPath = f'{predictions_dir}/{data_pre}_DNN_{transform}_prediction.csv'
    df_pred.to_csv(resultCSVPath, index=False, na_rep=0)

if __name__=='__main__':
    main()