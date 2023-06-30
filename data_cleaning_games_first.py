import pandas as pd

games = pd.read_csv('../data/raw/game_detials_0422/games.csv')

# drop missing data
games.dropna(axis=0, inplace=True)

# drop duplicates
games.drop_duplicates(subset=['GAME_ID'], inplace=True)

# save to csv
games.to_csv('../data/processing/cleaned_games.csv', index=False)