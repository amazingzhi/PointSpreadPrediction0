import pandas as pd

players = pd.read_csv('../data/processing/cleaned_games_details_after_EDA.csv')

new_data = players.sort_values(by=['year', 'month', 'day', 'GAME_ID', 'TEAM_ID','START_POSITION'],
                               ascending=[False, False, False, False, False, True])
new_data.to_csv('../data/processing/cleaned_games_details_after_EDA_sorted_time.csv', index=False)

