import pandas as pd

df = pd.read_csv('../data/final/final_game_features.csv')

pts_col_drop = ['oppo_lag1_pointspread','oppo_lag2_pointspread','oppo_lag3_pointspread',
                              'oppo_lag4_pointspread','oppo_lag5_pointspread','oppo_lag6_pointspread',
                              'oppo_lag7_pointspread','oppo_lag8_pointspread','oppo_lag9_pointspread',
                              'oppo_lag10_pointspread','oppo_lag11_pointspread','oppo_lag12_pointspread',
                              'oppo_lag13_pointspread', 'team_lag1_pointspread','team_lag2_pointspread',
                              'team_lag3_pointspread',
                              'team_lag4_pointspread','team_lag5_pointspread','team_lag6_pointspread',
                              'team_lag7_pointspread','team_lag8_pointspread','team_lag9_pointspread',
                              'team_lag10_pointspread','team_lag11_pointspread','team_lag12_pointspread',
                              'team_lag13_pointspread', 'vs_oppo1_pointspread', 'vs_oppo2_pointspread']

pts_df = df.drop(pts_col_drop, axis=1)

ps_col_keep = [i.split('_')[0] + '_' + i.split('_')[1] + '_' + 'loc' for i in pts_col_drop] + pts_col_drop + \
              ['vs_oppo1_FIC', 'vs_oppo2_FIC']
ps_df_keep = df[ps_col_keep]
ps_col_drop = df.columns[df.columns.str.contains('lag|vs_oppo')]
ps_df = df.drop(ps_col_drop, axis=1)
ps_df = ps_df.merge(ps_df_keep, left_index=True, right_index=True)
for i in pts_col_drop:
    temp = i.split('_')[0] + '_' + i.split('_')[1] + '_'
    ps_df[f'{temp}locps'] = ps_df[i] * ps_df[f'{temp}loc']
    pts_df[f'{temp}locpts'] = pts_df[f'{temp}PTS'] * pts_df[f'{temp}loc']

ps_df.to_csv('../data/final/final_game_features_ps.csv', index=False)
pts_df.to_csv('../data/final/final_game_features_pts.csv', index=False)
