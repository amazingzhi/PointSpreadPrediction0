import pandas as pd

def add_oppo_pred(row, df):
    game_id = row['GAME_ID']
    oppo_id = row['oppo_id']
    found_line = df[(df['GAME_ID'] == game_id) & (df['team_id'] == oppo_id)]
    oppo_pred_pts, oppo_pred_pms = found_line['team_PTS_pred'], found_line['team_PMS_pred']
    return list(oppo_pred_pts)[0], list(oppo_pred_pms)[0]
pred = pd.read_csv('../predictions/predictions_all_test/feature_selection_SGD_standard_prediction.csv')
group_by_gid_tid = pred.groupby(['GAME_ID','TEAM_ID'], as_index=False).sum()
group_by_gid_tid = group_by_gid_tid.rename(columns={'TEAM_ID': 'team_id', 'PTS_pred': 'team_PTS_pred',
                                                    'PLUS_MINUS_pred': 'team_PMS_pred'})
group_by_gid_tid = group_by_gid_tid[['GAME_ID', 'team_id', 'team_PTS_pred', 'team_PMS_pred']]

df = pd.read_csv('../data/processing/game_features_after_DEA.csv')
df = df.astype({'GAME_ID':str})
df = df[~df['GAME_ID'].str.startswith('1')]

group_by_gid_tid = group_by_gid_tid.astype({'GAME_ID':str})
df = df.merge(group_by_gid_tid, how='inner', on=['GAME_ID', 'team_id'])
df['oppo_PTS_pred'], df['oppo_PMS_pred'] = zip(*df.apply(lambda row: add_oppo_pred(row, df), axis=1))

team_oppo = ['team', 'oppo']
let_cols_to_drop_ps = ['oppo_let_avg_EFG%', 'oppo_let_avg_PPS', 'oppo_let_avg_PTS', 'oppo_let_avg_FG3A',
                       'oppo_let_avg_FG3M', 'oppo_let_avg_FGM', 'oppo_let_avg_FGA', 'oppo_let_avg_FIC']
let_cols_to_drop_pts = ['oppo_let_avg_pointspread', 'oppo_let_avg_PLUS', 'oppo_let_avg_FG3M', 'oppo_let_avg_EFG%',
                        'oppo_let_avg_PPS', 'oppo_let_avg_FIC']
rol_avg_cols_to_drop_ps = ['team_rol_avg_73_PPS', 'team_rol_avg_73_FG3M', 'team_rol_avg_73_EFG%', 'team_rol_avg_73_FG3A',
                           'team_rol_avg_73_PTS', 'team_rol_avg_73_FGM', 'team_rol_avg_73_FGA', 'team_rol_avg_73_FIC',
                           'team_rol_avg_73_DREB', 'team_rol_avg_73_FTA', 'team_rol_avg_73_REB', 'team_rol_avg_73_PF',
                           'oppo_rol_avg_73_PPS', 'oppo_rol_avg_73_FG3M', 'oppo_rol_avg_73_EFG%',
                           'oppo_rol_avg_73_FG3A',
                           'oppo_rol_avg_73_PTS', 'oppo_rol_avg_73_FGM', 'oppo_rol_avg_73_FGA', 'oppo_rol_avg_73_FIC',
                           'oppo_rol_avg_73_DREB', 'oppo_rol_avg_73_FTA', 'oppo_rol_avg_73_REB', 'oppo_rol_avg_73_PF'
                           ]
add_rol_avg_to_drop_ps = [col for col in df.columns if '82' in col or '33' in col or '23' in col]
rol_avg_cols_to_drop_ps = rol_avg_cols_to_drop_ps + add_rol_avg_to_drop_ps

rol_avg_cols_to_drop_pts = ['team_rol_avg_82_pointspread', 'team_rol_avg_82_FG', 'team_rol_avg_82_REB', 'team_rol_avg_82_FGA',
                            'team_rol_avg_82_FG3M', 'team_rol_avg_82_FG3A', 'team_rol_avg_82_FTM', 'team_rol_avg_82_FTA',
                            'team_rol_avg_82_DREB', 'team_rol_avg_82_PLUS', 'team_rol_avg_82_EFG%', 'team_rol_avg_82_PPS',
                            'team_rol_avg_82_FIC', 'oppo_rol_avg_82_pointspread', 'oppo_rol_avg_82_FG', 'oppo_rol_avg_82_REB', 'oppo_rol_avg_82_FGA',
                            'oppo_rol_avg_82_FG3M', 'oppo_rol_avg_82_FG3A', 'oppo_rol_avg_82_FTM', 'oppo_rol_avg_82_FTA',
                            'oppo_rol_avg_82_DREB', 'oppo_rol_avg_82_PLUS', 'oppo_rol_avg_82_EFG%', 'oppo_rol_avg_82_PPS',
                            'oppo_rol_avg_82_FIC']
add_rol_avg_to_drop_pts = [col for col in df.columns if '73' in col or '33' in col or '23' in col]
rol_avg_cols_to_drop_pts = rol_avg_cols_to_drop_pts + add_rol_avg_to_drop_pts

vs_cols_to_drop_ps = ['vs_oppo1_FG3M', 'vs_oppo1_BLK', 'vs_oppo1_EFG%', 'vs_oppo1_FGM', 'vs_oppo1_FG3A', 'vs_oppo1_OREB',
                      'vs_oppo1_DREB', 'vs_oppo1_AST', 'vs_oppo1_FGA', 'vs_oppo1_FG3A', 'vs_oppo1_STL', 'vs_oppo1_TO',
                      'vs_oppo1_REB', 'vs_oppo1_PF', 'vs_oppo2_FG3M', 'vs_oppo2_BLK', 'vs_oppo2_EFG%', 'vs_oppo2_FGM',
                      'vs_oppo2_FG3A', 'vs_oppo2_OREB', 'vs_oppo2_DREB', 'vs_oppo2_AST', 'vs_oppo2_FGA',
                      'vs_oppo2_FG3A', 'vs_oppo2_STL', 'vs_oppo2_TO', 'vs_oppo2_REB', 'vs_oppo2_PF']
vs_cols_to_drop_pts = ['vs_oppo1_pointspread', 'vs_oppo1_FG', 'vs_oppo1_AST', 'vs_oppo1_FGM', 'vs_oppo1_FG3M',
                       'vs_oppo1_FG3A', 'vs_oppo1_FTM', 'vs_oppo1_FTA', 'vs_oppo1_OREB', 'vs_oppo1_DREB',
                       'vs_oppo1_STL', 'vs_oppo1_BLK', 'vs_oppo1_TO', 'vs_oppo1_PF', 'vs_oppo1_PLUS', 'vs_oppo1_EFG%',
                       'vs_oppo1_PPS', 'vs_oppo2_pointspread', 'vs_oppo2_FG', 'vs_oppo2_AST', 'vs_oppo2_FGM',
                       'vs_oppo2_FG3M', 'vs_oppo2_FG3A', 'vs_oppo2_FTM', 'vs_oppo2_FTA', 'vs_oppo2_OREB',
                       'vs_oppo2_DREB', 'vs_oppo2_STL', 'vs_oppo2_BLK', 'vs_oppo2_TO', 'vs_oppo2_PF', 'vs_oppo2_PLUS',
                       'vs_oppo2_EFG%','vs_oppo2_PPS']


lags_cols_part_to_drop_ps = ['FG3M', 'FG3M', 'FGM', 'FTM', 'EFG%', 'STL', 'PPS', 'FG3A', 'TO', 'PF', 'AST', 'DREB',
                             'OREB', 'BLK']
lags_cols_to_drop_ps = []
for i1 in team_oppo:
    for i2 in range(1,14):
        for i3 in lags_cols_part_to_drop_ps:
            lags_cols_to_drop_ps.append(f'{i1}_lag{i2}_{i3}')
lags_cols_part_to_drop_pts = ['pointspread', 'AST', 'REB', 'FGM', 'FG3M', 'FG3A', 'FTM', 'OREB', 'DREB', 'STL', 'BLK',
                             'TO', 'PF', 'PLUS', 'EFG%', 'PPS']
lags_cols_to_drop_pts = []
for i1 in team_oppo:
    for i2 in range(1,14):
        for i3 in lags_cols_part_to_drop_pts:
            lags_cols_to_drop_pts.append(f'{i1}_lag{i2}_{i3}')


cols_to_drop_ps = let_cols_to_drop_ps + lags_cols_to_drop_ps + rol_avg_cols_to_drop_ps + vs_cols_to_drop_ps
cols_to_drop_pts = let_cols_to_drop_pts + lags_cols_to_drop_pts + rol_avg_cols_to_drop_pts + vs_cols_to_drop_pts

df_ps = df.drop(columns=cols_to_drop_ps)
df_pts = df.drop(columns=cols_to_drop_pts)

df_ps.to_csv('../data/final/final_game_features_ps.csv', index=False)
df_pts.to_csv('../data/final/final_game_features_pts.csv', index=False)