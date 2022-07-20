import pandas as pd

file_path_0420_team = 'data/time_series_data_generation_purpose/complete_0420_team_data.csv'
file_path_train = 'data/time_series_data_generation_purpose/pca_train.csv'
column_to_keep_test_data = ['GAME_ID', 'Sum_H_PM', 'Sum_A_PM', 'Dif_HA_PM']
out_path = 'data/time_series_data_generation_purpose/time_series_data.csv'


class TimeSeriesDataGeneration:
    """

    """

    def __init__(self, *paths: str, column_to_keep_test_data: list[str],
                 out_path: str) -> 'TimeSeriesDataGeneration object':
        self.paths = paths
        self.column_to_keep_test_data = column_to_keep_test_data
        self.out_path = out_path

    # main processes functions
    def read_data(self, paths: tuple[str]) -> 'Pandas DataFrames':
        # special part
        team = pd.read_csv(paths[0])
        train = pd.read_csv(paths[1])
        return team, train

    def data_cleaning(self, data: 'Pandas DataFrames') -> 'Pandas DataFrames':
        # special part
        train = data[data['GAME_DATE_EST'].str.contains('2012|2013|2014|2015|2016|2017|2018')]
        train = train[self.column_to_keep_test_data]
        return train

    def get_complete_original_data(self, *data: 'Pandas DataFrames') -> list and list[list]:
        # special part
        team, train = data
        df = team.merge(train, how='inner', on='GAME_ID')
        columns = df.columns
        df_list = df.values.tolist()
        return columns, df_list

    def loop_all_rows_to_get_new_data(self, data: list[list]) -> list[list]:
        pass

    def list_to_pandas_with_ColumnNames(self, column_names: list, data: list[list]) -> 'Pandas':
        df = pd.DataFrame(data, columns=column_names)
        return df

    def save_to_csv(self, data: 'Pandas') -> None:
        out_path = self.out_path
        data.to_csv(out_path, index=False, na_rep=0)

    # supplementary functions
    def find_past_82_games_for_one_gameID(self, gameID: str, data: list[list]) -> list[list[float]]:
        symbol = gameID


if __name__ == '__main__':
    T = TimeSeriesDataGeneration(file_path_0420_team, file_path_train, column_to_keep_test_data=column_to_keep_test_data,
                                 out_path=out_path)
    team, train = T.read_data(T.paths)
    train = T.data_cleaning(train)
    columns, df_list = T.get_complete_original_data(team, train)
