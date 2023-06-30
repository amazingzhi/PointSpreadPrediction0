import csv
from configparser import ConfigParser
import numpy as np
import copy
import pandas as pd

# global variables
file = '../configs/feature_creation_player_data_non_time.ini'
config = ConfigParser()
config.read(file)
data_path = config['paths']['read_path']
out_path = config['paths']['save_path']
need_vs_oppo = False
need_lags = False
need_oppo_avg = True
need_rolling_avg = False
need_pure_time = True
list_of_nums_rol_avg = [13, 23, 33]
list_of_nums_oppo = [82]
oppo_avg_num = 82
lags_num = 13
vs_oppo_num = 1

def read_data(data_path):
    PlayerData = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            PlayerData.append(row)
    columns = PlayerData[0]
    PlayerData = PlayerData[1:]
    return columns, PlayerData

def get_vs_oppo(player_id, data, oppo_id, vs_oppo_num=1):
    """data starting from looping current line"""
    i = 0
    vs_oppo = []
    for game in data:
        if game[7] == player_id and game[4] == oppo_id:
            vs_oppo.append(game[9:23] + game[26:])
            i += 1
            if i > vs_oppo_num:
                break
    if i != vs_oppo_num + 1:
        return False
    else:
        vs_oppo_new = []
        for i in vs_oppo[1:]:
            vs_oppo_new += i
        return vs_oppo_new

def get_lags(player_id, data, n_lag=13):
    """data starting from looping current line"""
    i = 0
    lags = []
    for game in data:
        if game[7] == player_id:
            if i == 0:
                i += 1
                continue
            else:
                lags += game[9:23] + game[26:]
                i += 1
        if i > n_lag:
            break
    if i != n_lag+1:
        return False
    else:
        return lags

def get_rolling_avg_lags(player_id, data, list):
    """data starting from looping current line"""
    n_lag = max(list)
    i = 0
    lags = []
    for game in data:
        if game[7] == player_id:
            if i == 0:
                i += 1
                continue
            else:
                lags.append(game[9:22] + game[26:28])
                i += 1
        if i > n_lag:
            break
    if i != n_lag+1:
        return False
    else:
        mean = list_avg(lags, list)
        return mean


def get_oppo_player_avg(start_pos, loc, oppo_id, data, need_pure_time, n_lag=82):
    """data starting from looping current line"""
    gameids, stop_index = get_oppo_lags_gameid(oppo_id, data, n_lag)
    if not isinstance(gameids, list):
        if need_pure_time:
            return False, False
        else:
            return False
    else:
        games = []
        for game in data[:stop_index + 38]:
            if game[0] in gameids and game[4] == oppo_id and game[28:32] == start_pos and game[22] == loc:
                games.append(game[9:22] + game[26:28])
        mean = list_avg(games, list_of_nums_oppo)
        if need_pure_time:
            ori_data = data[0][9:22] + data[0][26:28]
            pure_data = list()
            for item1, item2 in zip(ori_data, mean):
                item = float(item1) - item2
                pure_data.append(item)
            return pure_data, mean
        else:
            return mean

def get_oppo_lags_gameid(oppo_id, data, n_lag=82):
    first_game_id = data[0][0]
    gameids = []
    for index, game in enumerate(data):
        if oppo_id == game[4]:
            gameids.append(game[0])
        if len(set(gameids)) == n_lag + 1:
            stop_index = index
            break
    gameids = set(gameids)
    gameids.remove(first_game_id)
    if len(gameids) != n_lag:
        return False, False
    else:
        return list(gameids), stop_index

def get_new_columns(old_columns, need_lags, need_oppo_avg, need_rolling_avg, need_vs_oppo, list_nums_rol_avg,
                    num_vs_oppo, need_pure_time):
    avg_82_col = []
    lags_col = []
    rol_avg_col = []
    vs_oppo_col = []
    pure_time_col = []
    if need_oppo_avg:
        for i in old_columns[9:22] + old_columns[26:28]:
            avg_82_col.append('avg_oppo_82_' + i)
    if need_lags:
        for i in range(1, lags_num + 1):
            for col in old_columns[9:23] + old_columns[26:]:
                lags_col.append('lag' + str(i) + '_' + col)
    if need_rolling_avg:
        list_nums_rol_avg.sort(reverse=True)
        for i in list_nums_rol_avg:
            for i1 in old_columns[9:22] + old_columns[26:28]:
                rol_avg_col.append(f'rol_avg_{str(i)}_{i1}')
    if need_vs_oppo:
        for i in range(1, num_vs_oppo + 1):
            for col in old_columns[9:23] + old_columns[26:]:
                vs_oppo_col.append('vs_oppo' + str(i) + '_' + col)
    if need_pure_time:
        for i in old_columns[9:22] + old_columns[26:28]:
            pure_time_col.append('pure' + i)
    return old_columns + avg_82_col + lags_col + rol_avg_col + vs_oppo_col + pure_time_col

def generate_need_data(need_lags, need_oppo_avg, need_rolling_avg, need_vs_oppo, PlayerData):
    new_data = []
    if need_lags and not need_oppo_avg and not need_rolling_avg:
        for index, game in enumerate(PlayerData):
            lags = get_lags(player_id=game[7], data=PlayerData[index:], n_lag=lags_num)
            if lags:
                new_data.append(game + lags)
        return new_data
    elif need_lags and need_oppo_avg and not need_rolling_avg:
        for index, game in enumerate(PlayerData):
            lags = get_lags(player_id=game[7], data=PlayerData[index:], n_lag=lags_num)
            if need_pure_time:
                pure_data, mean_oppo_82 = get_oppo_player_avg(start_pos=game[28:32], loc=game[22], oppo_id=game[4],
                                    data=PlayerData[index:], need_pure_time=need_pure_time,
                                    n_lag=oppo_avg_num)
            else:
                mean_oppo_82 = get_oppo_player_avg(start_pos=game[28:32], loc=game[22], oppo_id=game[4],
                                               data=PlayerData[index:], need_pure_time=need_pure_time,
                                               n_lag=oppo_avg_num)
            if lags and mean_oppo_82:
                if need_pure_time:
                    new_data.append(game + mean_oppo_82 + lags + pure_data)
                else:
                    new_data.append(game + mean_oppo_82 + lags)
        return new_data
    elif need_lags and need_oppo_avg and need_rolling_avg and not need_vs_oppo:
        for index, game in enumerate(PlayerData):
            lags = get_lags(player_id=game[7], data=PlayerData[index:], n_lag=lags_num)
            if need_pure_time:
                pure_data, mean_oppo_82 = get_oppo_player_avg(start_pos=game[28:32], loc=game[22], oppo_id=game[4],
                                                              data=PlayerData[index:], need_pure_time=need_pure_time,
                                                              n_lag=oppo_avg_num)
            else:
                mean_oppo_82 = get_oppo_player_avg(start_pos=game[28:32], loc=game[22], oppo_id=game[4],
                                                   data=PlayerData[index:], need_pure_time=need_pure_time,
                                                   n_lag=oppo_avg_num)
            rol_avgs = get_rolling_avg_lags(player_id=game[7], data=PlayerData[index:], list=list_of_nums_rol_avg)

            if lags and mean_oppo_82 and rol_avgs:
                if need_pure_time:
                    new_data.append(game + mean_oppo_82 + lags + rol_avgs + pure_data)
                else:
                    new_data.append(game + mean_oppo_82 + lags + rol_avgs)
        return new_data
    elif need_lags and not need_oppo_avg and need_rolling_avg:
        for index, game in enumerate(PlayerData):
            lags = get_lags(player_id=game[7], data=PlayerData[index:], n_lag=lags_num)
            rol_avgs = get_rolling_avg_lags(player_id=game[7], data=PlayerData[index:], list=list_of_nums_rol_avg)

            if lags and rol_avgs:
                new_data.append(game + lags + rol_avgs)
        return new_data
    elif not need_lags and need_oppo_avg and need_rolling_avg:
        for index, game in enumerate(PlayerData):
            if need_pure_time:
                pure_data, mean_oppo_82 = get_oppo_player_avg(start_pos=game[28:32], loc=game[22], oppo_id=game[4],
                                                              data=PlayerData[index:], need_pure_time=need_pure_time,
                                                              n_lag=oppo_avg_num)
            else:
                mean_oppo_82 = get_oppo_player_avg(start_pos=game[28:32], loc=game[22], oppo_id=game[4],
                                                   data=PlayerData[index:], need_pure_time=need_pure_time,
                                                   n_lag=oppo_avg_num)
            rol_avgs = get_rolling_avg_lags(player_id=game[7], data=PlayerData[index:], list=list_of_nums_rol_avg)

            if mean_oppo_82 and rol_avgs:
                if need_pure_time:
                    new_data.append(game + mean_oppo_82 + rol_avgs + pure_data)
                else:
                    new_data.append(game + mean_oppo_82 + rol_avgs)
        return new_data
    elif not need_lags and need_oppo_avg and not need_rolling_avg:
        for index, game in enumerate(PlayerData):
            if need_pure_time:
                pure_data, mean_oppo_82 = get_oppo_player_avg(start_pos=game[28:32], loc=game[22], oppo_id=game[4],
                                                              data=PlayerData[index:], need_pure_time=need_pure_time,
                                                              n_lag=oppo_avg_num)
            else:
                mean_oppo_82 = get_oppo_player_avg(start_pos=game[28:32], loc=game[22], oppo_id=game[4],
                                                   data=PlayerData[index:], need_pure_time=need_pure_time,
                                                   n_lag=oppo_avg_num)
            if mean_oppo_82:
                if need_pure_time:
                    new_data.append(game + mean_oppo_82 + pure_data)
                else:
                    new_data.append(game + mean_oppo_82)
        return new_data
    elif not need_lags and not need_oppo_avg and need_rolling_avg:
        for index, game in enumerate(PlayerData):
            rol_avgs = get_rolling_avg_lags(player_id=game[7], data=PlayerData[index:], list=list_of_nums_rol_avg)

            if rol_avgs:
                new_data.append(game + rol_avgs)
        return new_data
    elif need_lags and need_oppo_avg and need_rolling_avg and need_vs_oppo:
        for index, game in enumerate(PlayerData):
            lags = get_lags(player_id=game[7], data=PlayerData[index:], n_lag=lags_num)
            if need_pure_time:
                pure_data, mean_oppo_82 = get_oppo_player_avg(start_pos=game[28:32], loc=game[22], oppo_id=game[4],
                                                              data=PlayerData[index:], need_pure_time=need_pure_time,
                                                              n_lag=oppo_avg_num)
            else:
                mean_oppo_82 = get_oppo_player_avg(start_pos=game[28:32], loc=game[22], oppo_id=game[4],
                                                   data=PlayerData[index:], need_pure_time=need_pure_time,
                                                   n_lag=oppo_avg_num)
            rol_avgs = get_rolling_avg_lags(player_id=game[7], data=PlayerData[index:], list=list_of_nums_rol_avg)
            vs_oppo = get_vs_oppo(player_id=game[7], data=PlayerData[index:], oppo_id=game[4], vs_oppo_num=vs_oppo_num)

            if lags and mean_oppo_82 and rol_avgs and vs_oppo:
                if need_pure_time:
                    new_data.append(game + mean_oppo_82 + lags + rol_avgs + vs_oppo + pure_data)
                else:
                    new_data.append(game + mean_oppo_82 + lags + rol_avgs + vs_oppo)
        return new_data

def list_avg(list_2d, list):
    list.sort(reverse=True)
    numpy_data = np.array(list_2d)
    numpy_data = numpy_data.astype(np.float)
    means = []
    for i in list:
        if i != max(list):
            mean = np.mean(numpy_data[:i+1, :], axis=0)
        else:
            mean = np.mean(numpy_data, axis=0)
        mean = mean.tolist()
        means.extend(mean)
    return means

def main():
    old_columns, PlayerData = read_data(data_path)
    PlayerDataCopy = copy.deepcopy(PlayerData)
    new_col = get_new_columns(old_columns, need_lags=need_lags, need_oppo_avg=need_oppo_avg,
                              need_rolling_avg=need_rolling_avg, need_vs_oppo=need_vs_oppo,
                              list_nums_rol_avg=list_of_nums_rol_avg, num_vs_oppo=vs_oppo_num, need_pure_time=need_pure_time)
    new_data = generate_need_data(need_lags, need_oppo_avg, need_rolling_avg, need_vs_oppo, PlayerDataCopy)

    df = pd.DataFrame(new_data, columns=new_col)
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()