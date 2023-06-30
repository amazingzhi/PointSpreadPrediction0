import pandas as pd
import numpy as np
from configparser import ConfigParser

# global variables
file = '../configs/data_cleaning_after_EDA.ini'
config = ConfigParser()
config.read(file)
read_path_player = config['paths']['read_path_player']
read_path_game = config['paths']['read_path_game']
save_path = config['paths']['save_path']

# main function
def main():
    df = pd.read_csv(read_path_player)
    # drop not important columns
    df.drop(['PF', 'TO', 'BLK', 'STL', 'OREB'], axis=1, inplace=True)
    # add pre, regular, and post season
    df['season'] = df['GAME_ID'].apply(season_creation)
    # add time and location information
    df_game = pd.read_csv(read_path_game)
    df_game = df_game[['GAME_DATE_EST', 'GAME_ID', 'HOME_TEAM_ID']]
    df_new = df.merge(df_game, how='left', on='GAME_ID')
    df_new['loc'] = df_new.apply(lambda row: 1 if row['TEAM_ID'] == row['HOME_TEAM_ID'] else 0, axis=1)
    df_new['year'] = df_new.apply(lambda row: row['GAME_DATE_EST'].split('-')[0], axis=1)
    df_new['month'] = df_new.apply(lambda row: row['GAME_DATE_EST'].split('-')[1], axis=1)
    df_new['day'] = df_new.apply(lambda row: row['GAME_DATE_EST'].split('-')[2], axis=1)
    df_new.to_csv(save_path, index=False)
# other functions
def season_creation(integer):
    if str(integer)[0] == '1':
        return 1
    elif str(integer)[0] == '2':
        return 2
    else:
        return 3
if __name__ == '__main__':
    main()