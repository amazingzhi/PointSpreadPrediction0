import pandas as pd

df = pd.read_csv('../data/processing/cleaned_games_details_after_EDA_final.csv')

df.replace({'START_POSITION': {1: 'F', 2: 'C', 3: 'G', 4: 'B'}, 'season': {1: 'Pre', 2: 'Reg', 3: 'Pos'}}, inplace=True)
df1 = pd.get_dummies(data=df, columns=['START_POSITION', 'season'])

df1.to_csv('../data/processing/cleaned_games_details_after_EDA_final.csv', index=False)