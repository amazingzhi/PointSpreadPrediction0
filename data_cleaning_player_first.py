# import packages
import pandas as pd
from configparser import ConfigParser
import statsmodels.api as sm
# global variables
file = '../configs/data_cleaning.ini'
config = ConfigParser()
config.read(file)
read_path = config['paths']['read_path']
save_path = config['paths']['save_path']
# functions
def main():
    # read row data
    df = pd.read_csv(read_path)
    # change start position nan values to bench
    df['START_POSITION'] = df['START_POSITION'].fillna('bench')
    # change position to numbers
    mapping = {'F': 1, 'C': 2, 'G': 3, 'bench': 4}
    df["START_POSITION"].replace(mapping, inplace=True)
    # drop nan values of PM column
    df = df[df['PLUS_MINUS'].notna()]
    # drop nicknames
    df.drop(columns=['NICKNAME', 'COMMENT'], inplace=True)
    # Check for NaN under an entire DataFrame:
    df.isnull().values.any()
    # change MIN column
    df['MIN'] = df['MIN'].apply(change_MIN_column)
    # save data
    df.to_csv(save_path, index = False)
def change_MIN_column(string):
    if ':' in string:
        seconds = int(string.split(':')[0]) * 60 + int(string.split(':')[1])
        return seconds
    else:
        seconds = float(string) * 48 * 60
        return seconds

if __name__ == '__main__':
    main()