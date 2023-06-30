import pandas as pd
import statsmodels.api as sm
from src import example_web as ew

def add_oppo_pred(row, df):
    game_id = row['GAME_ID']
    oppo_id = row['oppo_id']
    found_line = df[(df['GAME_ID'] == game_id) & (df['team_id'] == oppo_id)]
    oppo_pred_pts, oppo_pred_pms = found_line['team_PTS_pred'], found_line['team_PMS_pred']
    return list(oppo_pred_pts)[0], list(oppo_pred_pms)[0]
pred = pd.read_csv('../predictions/predictions_all_test/feature_selection_SGD_standard_prediction.csv')
group_by_gid_tid = pred.groupby(['GAME_ID','TEAM_ID'], as_index=False).sum()
result_matrix_PTS = ew.calculate_metrics_true_pred(group_by_gid_tid['PTS'], group_by_gid_tid['PTS_pred'])
result_matrix_PS = ew.calculate_metrics_true_pred(group_by_gid_tid['PLUS_MINUS'], group_by_gid_tid['PLUS_MINUS_pred'])
corr_pts = group_by_gid_tid['PTS'].corr(group_by_gid_tid['PTS_pred'])
corr_pms = group_by_gid_tid['PLUS_MINUS'].corr(group_by_gid_tid['PLUS_MINUS_pred'])

games_df = pd.read_csv('../data/processing/game_features_after_DEA_second.csv')
group_by_gid_tid = group_by_gid_tid.rename(columns={'TEAM_ID': 'team_id', 'PTS_pred': 'team_PTS_pred',
                                                    'PLUS_MINUS_pred': 'team_PMS_pred'})
group_by_gid_tid = group_by_gid_tid[['GAME_ID', 'team_id', 'team_PTS_pred', 'team_PMS_pred']]
new_data = games_df.merge(group_by_gid_tid, how='inner', on=['GAME_ID', 'team_id'])
new_data['oppo_PTS_pred'], new_data['oppo_PMS_pred'] = zip(*new_data.apply(lambda row: add_oppo_pred(row, new_data), axis=1))
new_data.to_csv('../data/final/final_game_features.csv', index=False)
