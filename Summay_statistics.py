import pandas as pd

# making data frame
betting = pd.read_csv('data/betting_line_merged.csv')

# removing null values to avoid errors
betting.dropna(inplace=True)

# percentile list
perc = [.20, .40, .60, .80]

# list of dtypes to include
include = ['object', 'float', 'int']

# calling describe method
desc = betting.describe(percentiles=perc, include=include)

# display
print(desc)