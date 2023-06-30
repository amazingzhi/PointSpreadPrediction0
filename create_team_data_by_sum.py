import csv
import numpy as np
import pandas as pd


def read_data():
    PlayerData = []
    with open('../data/processing/cleaned_games_details.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            PlayerData.append(row)
    Player_cols = PlayerData[0]
    PlayerData = PlayerData[1:]
    Games = []
    with open('../data/raw/game_detials_0422/games.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            Games.append(row)
    Games_cols = Games[0]
    Games = Games[1:]
    new_columns = create_columns(Games_cols, Player_cols)
    NewData = []
    for i,v in enumerate(Games):
        gameid = v[1]
        homeid = v[3]
        awayid = v[4]
        OneObservation = generate_one_observation(gameid,homeid,awayid,PlayerData)
        if OneObservation:
            v.extend(OneObservation)
            NewData.append(v)
    df = pd.DataFrame(NewData, columns=new_columns)
    df['EFG%_home'], df['EFG%_away'] = list(zip(*df.apply(lambda row: calculate_EFG(row), axis=1)))
    df['PPS_home'], df['PPS_away'] = list(zip(*df.apply(lambda row: calculate_PPS(row), axis=1)))
    df['FIC_home'], df['FIC_away'] = list(zip(*df.apply(lambda row: calculate_FIC(row), axis=1)))
    df['pointspread'] = df.apply(lambda row: calculate_pointspread(row), axis=1)
    df.to_csv('../data/processing/complete_team_data_after_sum_player.csv')

def create_columns(columns_ori_game, columns_ori_player):
    new_added_cols = columns_ori_player[8:10] + columns_ori_player[11:13] + columns_ori_player[14:16] + \
           columns_ori_player[17:19] + columns_ori_player[21:25] + [columns_ori_player[26]]
    new_cols = columns_ori_game
    for h_a in ['home', 'away']:
        for i in new_added_cols:
            new_cols.append(i + '_' + h_a)
    return new_cols

def generate_one_observation(gameid,homeid,awayid,PlayerData):

    OneObservation_H = []
    OneObservation_A = []
    for index, value in enumerate(PlayerData):
        if value[0] == gameid:
            if value[1] == homeid:
                OneLine_H = value[8:10]
                OneLine_H.extend(value[11:13])
                OneLine_H.extend(value[14:16])
                OneLine_H.extend(value[17:19])
                OneLine_H.extend(value[21:25])
                OneLine_H += [value[26]]
                OneObservation_H.append(OneLine_H)
            elif value[1] == awayid:
                OneLine_A = value[8:10]
                OneLine_A.extend(value[11:13])
                OneLine_A.extend(value[14:16])
                OneLine_A.extend(value[17:19])
                OneLine_A.extend(value[21:25])
                OneLine_A += [value[26]]
                OneObservation_A.append(OneLine_A)
            else:
                print(f'error, this {value[0]} game_id cannot find team_id.')
    if not OneObservation_H or not OneObservation_A:
        print(f'cannot find game_id {gameid}.')
        return False
    else:
        OneObservation_H = sum(OneObservation_H)
        OneObservation_A = sum(OneObservation_A)
        OneObservation_H.extend(OneObservation_A)
        return OneObservation_H

def calculate_EFG(row):
    row = pd.to_numeric(row, errors='coerce')
    return (row['FGM_home'] - row['FG3M_home'] + 1.5*row['FG3M_home']) / row['FGA_home'], \
           (row['FGM_away'] - row['FG3M_away'] + 1.5*row['FG3M_away']) / row['FGA_away']

def calculate_PPS(row):
    row = pd.to_numeric(row, errors='coerce')
    return (3*row['FG3M_home'] + 2*(row['FGM_home'] - row['FG3M_home']))/row['FGA_home'], \
           (3*row['FG3M_away'] + 2*(row['FGM_away'] - row['FG3M_away']))/row['FGA_away']

def calculate_FIC(row):
    row = pd.to_numeric(row, errors='coerce')
    return (row['PTS_home'] + row['OREB_home'] + 0.75*row['DREB_home'] + row['AST_home'] + row['STL_home'] + row['BLK_home'] - 0.75*row['FGA_home'] - 0.375*row['FTA_home'] - row['TO_home'] - 0.5*row['PF_home'], row['PTS_away'] + row['OREB_away'] + 0.75*row['DREB_away'] + row['AST_away'] + row['STL_away'] + row['BLK_away'] - 0.75*row['FGA_away'] - 0.375*row['FTA_away'] - row['TO_away'] - 0.5*row['PF_away'])

def calculate_pointspread(row):
    row = pd.to_numeric(row, errors='coerce')
    return row['PTS_home'] - row['PTS_away']

def sum(Alist):
    ndarray = np.array(Alist)
    ndarray = ndarray.astype(float)
    sumed_np = np.sum(ndarray, axis=0)
    sumed_np = sumed_np.tolist()
    sumed_np = list(map(str, sumed_np))
    return sumed_np
if __name__ == "__main__":
    read_data()
