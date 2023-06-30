import pandas as pd
import numpy as np
from configparser import ConfigParser
import json

# global variables
file = '../configs/feature_creation_player_data.ini'
config = ConfigParser()
config.read(file)
data_path = config['paths']['read_path']
oppo_avg_num = json.loads(config.get('nums','oppo_avg_num'))

def main():
    # read data
    df = pd.read_csv(data_path)
    # get data for PTS
    non_time = get_data(df=df.copy(), target='PTS')
    # save data
    # time.to_csv(config['paths']['save_path_time_PTS'], index=False)
    non_time.to_csv(config['paths']['save_path_non_time_PTS'], index=False)

def get_data(df, target):
    # df_non_time = df.iloc[:, np.r_[0:5, 7, 9, 23:30]]
    columns_mean = []
    for num in oppo_avg_num:
        for name in df.columns[np.r_[10:23, 28:30]]:
            columns_mean.append('mean_' + str(num) + '_' + name)

    list_of_means = []
    # list_of_pure_time = []
    indexes = []
    for index, row in df.iterrows():
        gameids = find_oppo_past_82_gameid_by_given_oppoid(row['OPPO_ID'], df.loc[index:, ['GAME_ID', 'OPPO_ID', 'GAME_DATE_EST']])
        if not isinstance(gameids, (np.ndarray, np.generic)):
            continue
        else:
            indexes.append(index)
            row_mean = []
            for num in oppo_avg_num:
                gameids_temp = gameids[:num]
                df_temp = df.loc[df['GAME_ID'].isin(gameids_temp)]
                df_temp = df_temp.loc[df_temp['TEAM_ID'] != row['OPPO_ID']]
                df_temp = df_temp.loc[(df_temp['START_POSITION'] == row['START_POSITION']) & (df_temp['loc'] == row['loc'])]
                df_temp = df_temp.iloc[:, np.r_[10:23, 28:30]]
                row_mean += list(pandas_calculate_average(df_temp))
            list_of_means.append(row_mean)
            # row_time = row.iloc[np.r_[10:23, 28:30]].to_numpy()
            # row_time = row_time - row_mean
            # list_of_pure_time.append(list(row_time))
    df_of_means = pd.DataFrame(list_of_means, index=indexes, columns=columns_mean)
    # df_of_pure_time = pd.DataFrame(list_of_pure_time, index=indexes, columns=df.columns[np.r_[10:23, 28:30]])
    df_non_time_final = df.merge(df_of_means, how='right', left_index=True, right_index=True)
    # df_time = df_of_pure_time.merge(df_non_time, how='left', left_index=True, right_index=True)
    # df_time = df_time.sort_values(['PLAYER_ID'])
    return df_non_time_final

# find opppo past 82 games id by given oppoid
def find_oppo_past_82_gameid_by_given_oppoid(oppoid, data): # data is siliced from current line and only include gameid, oppoid, datetime.
    data = data[data['OPPO_ID'] == oppoid]
    temp = pd.unique(data['GAME_ID'])
    if len(temp) <= 82:
        return False
    elif len(temp) == 83:
        return temp[1:]
    else:
        return temp[1:83]

def pandas_calculate_average(df):
    data = df.to_numpy()
    mean = np.mean(data, axis=0)
    return mean

if __name__ == '__main__':
    main()