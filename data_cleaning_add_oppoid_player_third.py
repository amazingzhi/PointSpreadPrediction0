import csv
Players = []
with open('../data/processing/cleaned_games_details_after_EDA_sorted_time.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        Players.append([row[0], row[1]])
Players = Players[1:]
Games = {}
with open('../data/processing/cleaned_games.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        Games[row[1]] = row[3:5]
oppoids = []
for i in Players:
    if i[1] == Games[i[0]][0]:
        oppoids.append(Games[i[0]][1])
    else:
        oppoids.append(Games[i[0]][0])
import pandas as pd

df = pd.read_csv('../data/processing/cleaned_games_details_after_EDA_sorted_time.csv')
df['OPPO_ID'] = oppoids
df = df.loc[:,['GAME_ID','GAME_DATE_EST','HOME_TEAM_ID','TEAM_ID','OPPO_ID','TEAM_ABBREVIATION','TEAM_CITY',
                'PLAYER_ID','PLAYER_NAME','START_POSITION','MIN','FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM',
                'FTA','FT_PCT','DREB','REB','AST','season','loc','year','month','day','PTS','PLUS_MINUS']]
df.to_csv('../data/processing/cleaned_games_details_after_EDA_final.csv')