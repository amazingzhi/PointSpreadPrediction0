import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#upload dataset
df_ps = pd.read_csv('../data/final/final_game_features_ps.csv')
df_pts = pd.read_csv('../data/final/final_game_features_pts.csv')
rest = df_ps.loc[:,:'FIC_oppo']
rest = pd.concat([rest, df_ps.loc[:,'team_PTS_pred':'oppo_PMS_pred']], axis=1)

# ps global variables
ps_oppo_let = df_ps.loc[:,'oppo_let_avg_pointspread':'oppo_let_avg_PLUS']
ps_team_lag1 = df_ps.loc[:,'team_lag1_pointspread':'team_lag1_FIC']
ps_team_lag2 = df_ps.loc[:,'team_lag2_pointspread':'team_lag2_FIC']
ps_team_lag3 = df_ps.loc[:,'team_lag3_pointspread':'team_lag3_FIC']
ps_team_lag4 = df_ps.loc[:,'team_lag4_pointspread':'team_lag4_FIC']
ps_team_lag5 = df_ps.loc[:,'team_lag5_pointspread':'team_lag5_FIC']
ps_team_lag6 = df_ps.loc[:,'team_lag6_pointspread':'team_lag6_FIC']
ps_team_lag7 = df_ps.loc[:,'team_lag7_pointspread':'team_lag7_FIC']
ps_team_lag8 = df_ps.loc[:,'team_lag8_pointspread':'team_lag8_FIC']
ps_team_lag9 = df_ps.loc[:,'team_lag9_pointspread':'team_lag9_FIC']
ps_team_lag10 = df_ps.loc[:,'team_lag10_pointspread':'team_lag10_FIC']
ps_team_lag11 = df_ps.loc[:,'team_lag11_pointspread':'team_lag11_FIC']
ps_team_lag12 = df_ps.loc[:,'team_lag12_pointspread':'team_lag12_FIC']
ps_team_lag13 = df_ps.loc[:,'team_lag13_pointspread':'team_lag13_FIC']
ps_oppo_lag1 = df_ps.loc[:,'oppo_lag1_pointspread':'oppo_lag1_FIC']
ps_oppo_lag2 = df_ps.loc[:,'oppo_lag2_pointspread':'oppo_lag2_FIC']
ps_oppo_lag3 = df_ps.loc[:,'oppo_lag3_pointspread':'oppo_lag3_FIC']
ps_oppo_lag4 = df_ps.loc[:,'oppo_lag4_pointspread':'oppo_lag4_FIC']
ps_oppo_lag5 = df_ps.loc[:,'oppo_lag5_pointspread':'oppo_lag5_FIC']
ps_oppo_lag6 = df_ps.loc[:,'oppo_lag6_pointspread':'oppo_lag6_FIC']
ps_oppo_lag7 = df_ps.loc[:,'oppo_lag7_pointspread':'oppo_lag7_FIC']
ps_oppo_lag8 = df_ps.loc[:,'oppo_lag8_pointspread':'oppo_lag8_FIC']
ps_oppo_lag9 = df_ps.loc[:,'oppo_lag9_pointspread':'oppo_lag9_FIC']
ps_oppo_lag10 = df_ps.loc[:,'oppo_lag10_pointspread':'oppo_lag10_FIC']
ps_oppo_lag11 = df_ps.loc[:,'oppo_lag11_pointspread':'oppo_lag11_FIC']
ps_oppo_lag12 = df_ps.loc[:,'oppo_lag12_pointspread':'oppo_lag12_FIC']
ps_oppo_lag13 = df_ps.loc[:,'oppo_lag13_pointspread':'oppo_lag13_FIC']
ps_team_rol = df_ps.loc[:,'team_rol_avg_73_pointspread':'team_rol_avg_73_PLUS']
ps_oppo_rol = df_ps.loc[:,'oppo_rol_avg_73_pointspread':'oppo_rol_avg_73_PLUS']
ps_vs1 = df_ps.loc[:,'vs_oppo1_pointspread':'vs_oppo1_FIC']
ps_vs2 = df_ps.loc[:,'vs_oppo2_pointspread':'vs_oppo2_FIC']
data_ps = [ps_oppo_let,ps_team_lag1,ps_team_lag2,ps_team_lag3,ps_team_lag4,ps_team_lag5,ps_team_lag6,ps_team_lag7,
           ps_team_lag8,ps_team_lag9,ps_team_lag10,ps_team_lag11,ps_team_lag12,ps_team_lag13,ps_oppo_lag1,ps_oppo_lag2,
           ps_oppo_lag3,ps_oppo_lag4,ps_oppo_lag5,ps_oppo_lag6,ps_oppo_lag7,ps_oppo_lag8,ps_oppo_lag9,ps_oppo_lag10,
           ps_oppo_lag11,ps_oppo_lag12,ps_oppo_lag13,ps_team_rol,ps_oppo_rol,ps_vs1,ps_vs2]
nums_ps = [9,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,8,8,5,5]
col_names_ps = ['oppo_let','team_lag1','team_lag2','team_lag3','team_lag4','team_lag5','team_lag6','team_lag7',
                'team_lag8','team_lag9','team_lag10','team_lag11','team_lag12','team_lag13','oppo_lag1','oppo_lag2',
                'oppo_lag3','oppo_lag4','oppo_lag5','oppo_lag6','oppo_lag7','oppo_lag8','oppo_lag9','oppo_lag10',
                'oppo_lag11','oppo_lag12','oppo_lag13','team_rol','oppo_rol','vs1','vs2']

# pts global variables
pts_oppo_let = df_pts.loc[:,'oppo_let_avg_PTS':'oppo_let_avg_PF']
pts_team_lag1 = df_pts.loc[:,'team_lag1_loc':'oppo_lag1_FIC']
pts_team_lag2 = df_pts.loc[:,'team_lag2_loc':'team_lag2_FIC']
pts_team_lag3 = df_pts.loc[:,'team_lag3_loc':'team_lag3_FIC']
pts_team_lag4 = df_pts.loc[:,'team_lag4_loc':'team_lag4_FIC']
pts_team_lag5 = df_pts.loc[:,'team_lag5_loc':'team_lag5_FIC']
pts_team_lag6 = df_pts.loc[:,'team_lag6_loc':'team_lag6_FIC']
pts_team_lag7 = df_pts.loc[:,'team_lag7_loc':'team_lag7_FIC']
pts_team_lag8 = df_pts.loc[:,'team_lag8_loc':'team_lag8_FIC']
pts_team_lag9 = df_pts.loc[:,'team_lag9_loc':'team_lag9_FIC']
pts_team_lag10 = df_pts.loc[:,'team_lag10_loc':'team_lag10_FIC']
pts_team_lag11 = df_pts.loc[:,'team_lag11_loc':'team_lag11_FIC']
pts_team_lag12 = df_pts.loc[:,'team_lag12_loc':'team_lag12_FIC']
pts_team_lag13 = df_pts.loc[:,'team_lag13_loc':'team_lag13_FIC']
pts_oppo_lag1 = df_pts.loc[:,'oppo_lag1_loc':'oppo_lag1_FIC']
pts_oppo_lag2 = df_pts.loc[:,'oppo_lag2_loc':'oppo_lag2_FIC']
pts_oppo_lag3 = df_pts.loc[:,'oppo_lag3_loc':'oppo_lag3_FIC']
pts_oppo_lag4 = df_pts.loc[:,'oppo_lag4_loc':'oppo_lag4_FIC']
pts_oppo_lag5 = df_pts.loc[:,'oppo_lag5_loc':'oppo_lag5_FIC']
pts_oppo_lag6 = df_pts.loc[:,'oppo_lag6_loc':'oppo_lag6_FIC']
pts_oppo_lag7 = df_pts.loc[:,'oppo_lag7_loc':'oppo_lag7_FIC']
pts_oppo_lag8 = df_pts.loc[:,'oppo_lag8_loc':'oppo_lag8_FIC']
pts_oppo_lag9 = df_pts.loc[:,'oppo_lag9_loc':'oppo_lag9_FIC']
pts_oppo_lag10 = df_pts.loc[:,'oppo_lag10_loc':'oppo_lag10_FIC']
pts_oppo_lag11 = df_pts.loc[:,'oppo_lag11_loc':'oppo_lag11_FIC']
pts_oppo_lag12 = df_pts.loc[:,'oppo_lag12_loc':'oppo_lag12_FIC']
pts_oppo_lag13 = df_pts.loc[:,'oppo_lag13_loc':'oppo_lag13_FIC']
pts_team_rol = df_pts.loc[:,'team_rol_avg_82_PTS':'team_rol_avg_82_PF']
pts_oppo_rol = df_pts.loc[:,'oppo_rol_avg_82_PTS':'oppo_rol_avg_82_PF']
pts_vs1 = df_pts.loc[:,'vs_oppo1_loc':'vs_oppo1_FIC']
pts_vs2 = df_pts.loc[:,'vs_oppo2_loc':'vs_oppo2_FIC']
data_pts = [pts_oppo_let,pts_team_lag1,pts_team_lag2,pts_team_lag3,pts_team_lag4,pts_team_lag5,pts_team_lag6,pts_team_lag7,
           pts_team_lag8,pts_team_lag9,pts_team_lag10,pts_team_lag11,pts_team_lag12,pts_team_lag13,pts_oppo_lag1,pts_oppo_lag2,
           pts_oppo_lag3,pts_oppo_lag4,pts_oppo_lag5,pts_oppo_lag6,pts_oppo_lag7,pts_oppo_lag8,pts_oppo_lag9,pts_oppo_lag10,
           pts_oppo_lag11,pts_oppo_lag12,pts_oppo_lag13,pts_team_rol,pts_oppo_rol,pts_vs1,pts_vs2]
nums_pts = [10,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,8,8,6,6]

def get_pca(num, col_name, data, last_data):
    columns = []
    for j in range(1, num + 1):
        columns.append(f'{col_name}_{j}')
    sc = StandardScaler()
    std = sc.fit_transform(data)
    pca = PCA(n_components=num)
    Pcaed = pca.fit_transform(std)
    tempDF = pd.DataFrame(data=Pcaed, columns=columns)
    final_df = pd.concat([last_data, tempDF], axis=1)
    return final_df


for data, num, col_name in zip(data_ps, nums_ps, col_names_ps):
    if col_name == 'oppo_let':
        final_df_ps = get_pca(num, col_name, data, rest)
    else:
        final_df_ps = get_pca(num, col_name, data, final_df_ps)

for data, num, col_name in zip(data_pts, nums_pts, col_names_ps):
    if col_name == 'oppo_let':
        final_df_pts = get_pca(num, col_name, data, rest)
    else:
        final_df_pts = get_pca(num, col_name, data, final_df_pts)

# save data
resultCSVPath_ps = '../data/final/ps_pca.csv'
final_df_ps.to_csv(resultCSVPath_ps,index = False,na_rep = 0)
resultCSVPath_pts = '../data/final/pts_pca.csv'
final_df_pts.to_csv(resultCSVPath_pts,index = False,na_rep = 0)