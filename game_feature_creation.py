import csv
from configparser import ConfigParser
import numpy as np
import copy
import pandas as pd

# global variables
file = '../configs/game_feature_creation.ini'
config = ConfigParser()
config.read(file)
data_path = config['paths']['read_path']
out_path = config['paths']['save_path']
need_vs_oppo = True
need_team_lags = True
need_oppo_lags = True
need_team_avg = True
need_oppo_avg = True
need_oppo_let_oppo_avg = True
need_pure_time = False
list_of_nums_rol_avg = [23, 33, 73, 82]
list_of_nums_oppo = [32, 82]
oppo_avg_num = 82
lags_num = 13
vs_oppo_num = 2

def read_data(data_path):
    Games = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            Games.append(row)
    columns = Games[0]
    Games = Games[1:]
    Games = Games[::-1]
    return columns, Games

def get_new_cols(old_cols, need_vs_oppo, need_team_lags, need_oppo_lags, need_team_avg, need_oppo_avg,
                 need_oppo_let_oppo_avg, need_pure_time):
    oppo_let_oppo_avg_col = []
    team_lags_col = []
    oppo_lags_col = []
    team_rol_avg_col = []
    oppo_rol_avg_col = []
    vs_oppo_col = []
    pure_time_col = []
    old_cols_modified = []
    for i in old_cols[7:31]:
        old_cols_modified.append(i.split('_')[0])
    if need_oppo_let_oppo_avg:
        for i in [old_cols_modified[0]] + old_cols_modified[2:]:
            oppo_let_oppo_avg_col.append('oppo_let_avg_' + i)
    if need_team_lags:
        for i in range(1, lags_num + 1):
            for col in old_cols_modified:
                team_lags_col.append('team_lag' + str(i) + '_' + col)
    if need_oppo_lags:
        for i in range(1, lags_num + 1):
            for col in old_cols_modified:
                oppo_lags_col.append('oppo_lag' + str(i) + '_' + col)
    if need_team_avg:
        list_of_nums_rol_avg.sort(reverse=True)
        for i in list_of_nums_rol_avg:
            for i1 in [old_cols_modified[0]] + old_cols_modified[2:]:
                team_rol_avg_col.append(f'team_rol_avg_{str(i)}_{i1}')
    if need_oppo_avg:
        list_of_nums_rol_avg.sort(reverse=True)
        for i in list_of_nums_rol_avg:
            for i1 in [old_cols_modified[0]] + old_cols_modified[2:]:
                oppo_rol_avg_col.append(f'oppo_rol_avg_{str(i)}_{i1}')
    if need_vs_oppo:
        for i in range(1, vs_oppo_num + 1):
            for col in old_cols_modified:
                vs_oppo_col.append('vs_oppo' + str(i) + '_' + col)
    if need_pure_time:
        for i in [old_cols_modified[0]] + old_cols_modified[2:]:
            pure_time_col.append('pure_' + i)
    return old_cols + pure_time_col + oppo_let_oppo_avg_col + team_lags_col + oppo_lags_col + team_rol_avg_col +\
           oppo_rol_avg_col + vs_oppo_col


def get_lags(data, n_lag):
    """data starting from looping current line"""
    if len(data) <= n_lag:
        return []
    if data[0][2] != data[n_lag][2]:
        return []
    lags = []
    i = 0
    while i < n_lag:
        lags.append(data[i + 1][7:31])
        i += 1
    return lags

def get_oppo_lags(data, n_lag, game_id, oppo_id):
    """data starting from beginning"""
    for i, game in enumerate(data):
        if game[1] == game_id and game[2] == oppo_id:
            start_index = i
            break
    lags = get_lags(data[start_index:], n_lag)
    return lags

def get_pure_lags(mean, data_current_line):
    """data starting from beginning, this loc is oppo's loc"""
    data_current_nums = [data_current_line[7]] + data_current_line[9:31]
    sub = list()
    for i1, i2 in zip(data_current_nums, mean):
        sub.append(float(i1) - i2)
    return sub

def get_oppo_avg(data, list_n_lag, game_id, oppo_id, loc):
    """data starting from beginning, this loc is oppo's loc"""
    for i, game in enumerate(data):
        if game[1] == game_id and game[2] == oppo_id:
            start_index = i
            break
    means = get_avg(data[start_index:], list_n_lag, loc)
    return means

def get_oppo_let_oppo_avg(data, game_id, oppo_id, loc, n_lag):
    """data starting from beginning, this loc is oppo's loc"""
    for i, game in enumerate(data):
        if game[1] == game_id and game[2] == oppo_id:
            start_index = i
            break
    if len(data[start_index:]) <= n_lag:
        return []
    data = data[start_index:]
    if data[0][2] != data[n_lag][2]:
        return []
    lags = []
    i = 0
    while i < n_lag:
        if data[i + 1][8] == loc:
            lags.append([-float(data[i + 1][7])] + data[i + 1][31:])
        i += 1
    mean = np_avg(lags)
    return mean

def get_avg(data, list_n_lag, loc):
    """data starting from looping current line, loc is team's loc"""
    list_n_lag.sort(reverse=True)
    lags = get_lags(data, max(list_n_lag))
    if not lags:
        return []
    means = []
    for n_lag in list_n_lag:
        temp = []
        if n_lag == max(list_n_lag):
            for lag in lags:
                if lag[1] == loc:
                    temp.append([lag[0]] + lag[2:])
        else:
            for lag in lags[:n_lag]:
                if lag[1] == loc:
                    temp.append([lag[0]] + lag[2:])
        mean = np_avg(temp)
        means.extend(mean)
    return means

def np_avg(list):
    numpy_data = np.array(list)
    numpy_data = numpy_data.astype(np.float)
    mean = np.mean(numpy_data, axis=0)
    mean = mean.tolist()
    return mean

def get_vs_oppo(data, n_lag, oppo_id):
    """data starting from looping current line"""
    if len(data) <= 82:
        return []
    if data[0][2] != data[82][2]:
        return []
    vs_oppo = []
    i = 0
    for game in data[1:]:
        if oppo_id == game[3]:
            vs_oppo += game[7:31]
            i += 1
        if i == n_lag:
            break
    if i != n_lag:
        return []
    else:
        return vs_oppo

def generate_need_data(need_vs_oppo, need_team_lags, need_oppo_lags, need_team_avg, need_oppo_avg,
                 need_oppo_let_oppo_avg, need_pure_time, data):
    new_data = []
    for index, game in enumerate(data):
        oppo_loc = '1' if game[8] == '0' else '0'
        if need_vs_oppo:
            vs_oppo = get_vs_oppo(data[index:], vs_oppo_num, game[3])
        if need_oppo_let_oppo_avg:
            oppo_let_avg = get_oppo_let_oppo_avg(data,game_id=game[1],oppo_id=game[3],loc=oppo_loc,n_lag=oppo_avg_num)
        if need_team_avg:
            team_avg = get_avg(data[index:], list_n_lag=list_of_nums_rol_avg,loc=game[8])
        if need_oppo_avg:
            oppo_avg = get_oppo_avg(data,list_n_lag=list_of_nums_rol_avg,game_id=game[1],oppo_id=game[3],loc=oppo_loc)
        if need_team_lags:
            team_lags = get_lags(data[index:],n_lag=lags_num)
            temp = []
            for i in team_lags:
                temp += i
            team_lags = temp
        if need_oppo_lags:
            oppo_lags = get_oppo_lags(data,n_lag=lags_num,game_id=game[1],oppo_id=game[3])
            temp = []
            for i in oppo_lags:
                temp += i
            oppo_lags = temp
        if need_pure_time and oppo_let_avg:
            pure = get_pure_lags(oppo_let_avg, data_current_line=game)
        #todo: change this based on your need.
        if oppo_let_avg and vs_oppo and team_avg and oppo_avg and team_lags and oppo_lags:
            new_data.append(game + oppo_let_avg + team_lags + oppo_lags + team_avg +\
           oppo_avg + vs_oppo)
    return new_data



def main():
    old_columns, data = read_data(data_path)
    DataCopy = copy.deepcopy(data)
    new_col = get_new_cols(old_columns, need_vs_oppo, need_team_lags, need_oppo_lags, need_team_avg, need_oppo_avg,
                 need_oppo_let_oppo_avg, need_pure_time)
    new_data = generate_need_data(need_vs_oppo, need_team_lags, need_oppo_lags, need_team_avg, need_oppo_avg,
                 need_oppo_let_oppo_avg, need_pure_time, DataCopy)

    df = pd.DataFrame(new_data, columns=new_col)
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()