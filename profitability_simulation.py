# import libraries
import numpy as np
import pandas as pd
import os
import glob
import csv
import matplotlib.pyplot as plt

# global variables
prediction_path = 'predictions/best_predictions'
test_path = 'data/Original_data/test.csv'
ori_path = 'data/12-18_standard_data.csv'
bet_path = 'data/betting_line_merged.csv'
games_all = 600
games_one = 80
new_columns_TF = ['DNN_TF', 'DTR_TF', 'GBR_TF', 'LGBM_TF', 'LR_TF', 'RF_TF', 'SVM_TF']
new_columns_AS = ['DNN_AS', 'DTR_AS', 'GBR_AS', 'LGBM_AS', 'LR_AS', 'RF_AS', 'SVM_AS', 'home_AS', 'away_AS']
b = 0.9


class ProfitSimulation:
    """ProfitsSimulation is a a helper class that Simulates profits.
    Attributes:
            prediction_path: where you save all of your predictions in a file
            test_path: where you save your test dataset
            ori_path: where you save your original dataset
            bet_path: where you save your betting data
            games_all: how many games to select to simulate profits from all games
            games_one: how many games to select to simulate profits from one game per day
            new_columns_TF: column names for recording True or False for each algorithm
            new_columns_AS: column names for recording Accumulate Sum for each algorithm
            b: is the proportion of the bet gained with a win. E.g. If betting $10 on a 2-to-1 odds bet, (upon win you
                are returned $30, winning you $20), then {\displaystyle b=\$20/\$10=2.0}{\displaystyle b=\$20/\$10=2.0}.


    Methods:
            merge_all_predictions_to_a_DF: the predictions are csv files in a folder. Each file means predictions from
                one algorithm. So this method is for merge them into one Data Frame. If you already have a merged file,
                don't apply this method.
            data_cleaning: After I merge, I found two observations are 0s. So I remove these two observations. If you
                don't need to clean your data, don't apply this method.
            add_gameid_to_df: Because I need to match our prediction to the real game outcomes and betting lines, I have
                to give each prediction a game id so that it can be matched.
            import_betting_data_and_clean: import betting lines and drop irrelevant columns. And then remove observations
                that have same gameid which comes from other bookies. We only need one bookie for one game.
            merge_betting_and_prediction_by_gameid: match betting data and prediction data together by game ids.
            add_always_bet_home_away: add two columns which always bet home team or away team record as True or False
                for comparision with other algorithm.
            prediction: calculate if prediction is right or wrong and record as True or False.
            add_t_or_f_algorithms: loop over each algorithm and calculate if this algorithm for each prediction is right
                or wrong and record as True or False.
            best_bet: find best bet on each day for one algorithm.
            create_dic_records_best_bet: build a dictionary carries all best_bet for all algorithms.
            kelly_fraction: calculate kelly fraction by probability and odds.
            kelly_accumulative_sum: calculate one accumulative sum by kelly criterion.
            accumulative_sum: records all accumulative sums each day for one algorithm.
            add_accumulative_sum_to_df: add accumulative sums to Pandas or Dict.
            plot_accumulative_sum: plot accumulative sums.
            prediction_comparision_plot: plot all predictions with actual values.
        """

    def __init__(self, prediction_path: str, test_path: str, ori_path: str, bet_path: str, games_all: int,
                 games_one: int, new_columns_TF: list[str], new_columns_AS: list[str], b: float)\
            -> 'ProfitSimulation object':
        self.prediction_path = prediction_path
        self.test_path = test_path
        self.ori_path = ori_path
        self.bet_path = bet_path
        self.games_all = games_all
        self.games_one = games_one
        self.new_columns_TF = new_columns_TF
        self.new_columns_AS = new_columns_AS
        self.b = b

    def merge_all_predictions_to_a_DF(self) -> 'Pandas':
        all_files = glob.glob(os.path.join(self.prediction_path, "*.csv"))
        data_dic = {}
        for file in all_files:
            with open(file) as file_object:
                contents = file_object.read()
                data = contents.split('\n')
                temp = file.split('\\')[-1]
                column = temp.split('.')[0]
                data_dic[column] = data
        df = pd.DataFrame.from_dict(data_dic)
        return df

    def prediction_data_cleaning(self, df: 'Pandas') -> 'Pandas':
        df = df.drop([0])
        df = df.drop([1231])
        return df

    def add_gameid_to_df(self, df: 'Pandas') -> 'Pandas':
        test = []
        with open(self.test_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for line in csv_reader:
                test.append([line[0], line[2]])
        test = test[1:]
        ori = []
        with open(self.ori_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for line in csv_reader:
                if line[5] == '1':
                    ori.append([line[0], line[1], line[3]])
        for ind, line in enumerate(test):
            for line_ori in ori:
                if line[0] == line_ori[0] and line[1] == line_ori[1]:
                    test[ind].append(line_ori[2])
        gameid = []
        date = []
        for line in test:
            gameid.append(line[2])
            date.append(line[0])
        df = df.set_index(pd.Index(gameid))
        df['date'] = date
        return df

    def import_betting_data_and_clean(self) -> 'Pandas':
        betting = pd.read_csv(self.bet_path)
        betting = betting.drop(columns=['book_name', 'book_id', 'a_team_id', 'h_team_id',
                                        'spread2', 'price1', 'price2', 'odds1', 'odds2'])
        betting = betting.set_index('GAME_ID')
        betting.index = betting.index.map(str)
        betting = betting[~betting.index.duplicated(keep='first')]
        return betting

    def merge_betting_and_prediction_by_gameid(self, df: 'Pandas', betting: 'Pandas') -> 'Pandas':
        df = pd.merge(df, betting, left_index=True, right_index=True)
        # some extra data cleaning work to do
        date = df['date']
        df = df.drop(columns='date')
        df = df.astype('float')
        df['date'] = date
        return df

    def add_always_bet_home_away(self, df: 'Pandas') -> 'Pandas':
        temp_home = []
        temp_away = []
        for index, row in df.iterrows():
            if row['bet_h_or_a'] == 1:
                temp_home.append(True)
                temp_away.append(False)
            else:
                temp_home.append(False)
                temp_away.append(True)
        print(f'the percentage of Ture of bet_home is {temp_home[:games_all].count(True) / len(temp_home[:games_all])}')
        print(f'the percentage of Ture of bet_away is {temp_away[:games_all].count(True) / len(temp_away[:games_all])}')
        df['home_TF'] = temp_home
        df['away_TF'] = temp_away
        return df

    # calculate if prediction is right or wrong and record as 1 or 0.
    def prediction(self, row: 'row in Pandas', column: 'column in Pandas') -> bool:
        temp = column.split('_')
        algorithm = temp[0]
        if row[algorithm] - row['spread1'] < 0:
            h_or_a = 0
        else:
            h_or_a = 1
        if h_or_a == row['bet_h_or_a']:
            return True
        return False

    def add_t_or_f_algorithms(self, df: 'pandas') -> 'pandas':
        for column in self.new_columns_TF:
            temp = []
            for index, row in df.iterrows():
                temp.append(self.prediction(row, column))
            df[column] = temp
            print(f'the percentage of Ture of {column} is {temp[:games_all].count(True) / len(temp[:games_all])}')
        return df

    def best_bet(self, df: 'Pandas', column: str) -> list[bool]:
        temp = column.split('_')
        algorithm = temp[0]
        dates = df.date.unique()
        TF = []
        for date in dates:
            temp_data = {}
            for index, row in df.iterrows():
                df_date = row['date']
                if date == df_date:
                    temp_data[index] = abs(row[algorithm] - row['spread1'])
            max_key = max(temp_data, key=temp_data.get)
            TF.append(df.loc[max_key][algorithm + '_TF'])
        print(f'the percentage of Ture of {column} is {TF[:self.games_one].count(True) / len(TF[:self.games_one])}')
        return TF

    def create_dic_records_best_bet(self, df: 'pandas') -> dict:
        best_bet_of_the_day = [column + '_best' for column in self.new_columns_TF]
        dic_best = {}
        for column in best_bet_of_the_day:
            dic_best[column] = self.best_bet(df, column)
        dic_best['home_TF_best'] = df['home_TF'][:len(dic_best[column])].tolist()
        dic_best['away_TF_best'] = df['away_TF'][:len(dic_best[column])].tolist()
        return dic_best

    # kelly fraction calculation
    def kelly_fraction(self, p: float, b: float) -> float:
        f = p + (p - 1) / b
        return f

    def kelly_accumulative_sum(self, f: float, b: float, last_AS: float, TF: bool) -> float:
        if TF == True:
            this_AS = last_AS * f * b + last_AS
            return this_AS
        else:
            this_AS = last_AS * (1 - f)
            return this_AS

    def accumulative_sum(self, column: str, data: 'pandas' or dict, all_not_best: bool) -> list[float]:
        temp = column.split('_')
        if all_not_best == True:
            algorithm = temp[0] + '_TF'
            p = data[algorithm][:self.games_all].value_counts(normalize=True).array[0]
        else:
            algorithm = temp[0] + '_TF_best'
            p = data[algorithm][:self.games_one].count(True) / len(data[algorithm][:self.games_one])
        f = self.kelly_fraction(p, self.b)
        acc_sum = []
        for i in data[algorithm]:
            if not acc_sum:
                acc_sum.append(self.kelly_accumulative_sum(f, self.b, 100, i))
            else:
                acc_sum.append(self.kelly_accumulative_sum(f, self.b, acc_sum[-1], i))
        return acc_sum

    def add_accumulative_sum_to_df(self, data: 'Pandas' or dict, all_not_best: bool) -> 'Pandas' or dict:
        for column in self.new_columns_AS:
            data[column] = self.accumulative_sum(column, data, all_not_best)
        return data

    def plot_accumulative_sum(self, df: 'Pandas' or dict):
        AS = {}
        AS['Deep_Neural_Networks'] = df['DNN_AS']
        AS['Decision_Tree'] = df['DTR_AS']
        AS['Gradient_Boost'] = df['GBR_AS']
        AS['LightGBM'] = df['LGBM_AS']
        AS['Linear_Regression'] = df['LR_AS']
        AS['Random_Forest'] = df['RF_AS']
        AS['Support_Vector_Machine'] = df['SVM_AS']
        AS['Bet_Home'] = df['home_AS']
        # AS['Bet_Away'] = df['away_AS']

        if isinstance(df, dict):
            for name in AS.keys():
                if name != 'Bet_Home':
                    plt.plot(range(1, self.games_one + 1), AS[name][:self.games_one], label=name)
        else:
            for name in AS.keys():
                plt.plot(range(1, self.games_all + 1), AS[name][:self.games_all], label=name)
        plt.axhline(y=100, color='k', linestyle='solid')
        plt.title('Profits Simulation')
        plt.xlabel('Number of Games Bet')
        plt.ylabel('Profits')
        plt.legend()
        plt.show()

    def prediction_comparision_plot(self, df: 'pandas'):
        PS = {}
        actual = df['pointspread(E-D)']
        PS['Deep_Neural_Networks'] = df['DNN']
        PS['Decision_Tree'] = df['DTR']
        PS['Gradient_Boost'] = df['GBR']
        PS['LightGBM'] = df['LGBM']
        PS['Linear_Regression'] = df['LR']
        PS['Random_Forest'] = df['RF']
        PS['Support_Vector_Machine'] = df['SVM']
        PS_keys = list(PS.keys())
        for i in range(2):
            fig, axs = plt.subplots(2, 2, sharex=True)
            axs_indexes = [[row, column] for row in range(2) for column in range(2)]
            for index, axs_index in enumerate(axs_indexes):
                axs[axs_index[0], axs_index[1]].plot(range(1, 688), actual, '-', label='Actual')
                if i == 0:
                    axs[axs_index[0], axs_index[1]].plot(range(1, 688), PS[PS_keys[index]], '-', label=PS_keys[index])
                    axs[axs_index[0], axs_index[1]].set_title(f'Actual vs {PS_keys[index]}')
                else:
                    if index != 3:
                        axs[axs_index[0], axs_index[1]].plot(range(1, 688), PS[PS_keys[index + 4]], '-', label=PS_keys[index + 4])
                        axs[axs_index[0], axs_index[1]].set_title(f'Actual vs {PS_keys[index + 4]}')
                    else:
                        break
                axs[axs_index[0], axs_index[1]].axhline(y=0, color='k', linestyle='solid')
                axs[axs_index[0], axs_index[1]].legend()

            for ax in axs.flat:
                ax.set(xlabel='Number of Games', ylabel='Point Spreads')
                ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots.
            if i == 1:
                fig.delaxes(axs[1,1])
            fig.show()

def main():
    # create profit simulation object
    S = ProfitSimulation(prediction_path, test_path, ori_path, bet_path, games_all, games_one, new_columns_TF,
                         new_columns_AS, b)
    # open all predictions files of test data from best_predictions folder and save them in a Pandas DataFrame.
    df = S.merge_all_predictions_to_a_DF()
    # data cleaning for this DataFrame
    df = S.prediction_data_cleaning(df)
    # add game id to this dataframe as index
    df = S.add_gameid_to_df(df)
    # open betting.csv and save it as a Pandas DataFrame
    betting = S.import_betting_data_and_clean()
    # merge betting and predictions to a new dataframe by their game ids. Lost some observations from test dataset
    # because missing values in betting data.
    df = S.merge_betting_and_prediction_by_gameid(df, betting)
    # plot prediction comparison
    S.prediction_comparision_plot(df)
    # add always bet home team algorithm as an traditional betting method.
    df = S.add_always_bet_home_away(df)
    # add columns to df that whether prediction is True or False for each algorithm.
    df = S.add_t_or_f_algorithms(df)
    # add columns to df that accounts for accumulative sum of the profits for each algorithm for all games in test
    # dataset.
    df = S.add_accumulative_sum_to_df(df, True)
    # plot accumulative sum of profits for all games in test dataset.
    S.plot_accumulative_sum(df)
    # build a dic to select the most deviate betting line each day as the best bet and save all days best bet into
    # this dic.
    dic_best = S.create_dic_records_best_bet(df)
    # add columns to df that accounts for accumulative sum of the profits for each algorithm for one game per day in
    # test dataset.
    dic_best = S.add_accumulative_sum_to_df(dic_best, False)
    # plot accumulative sum of profits for all games in test dataset.
    S.plot_accumulative_sum(dic_best)
    print('finished')


if __name__ == "__main__":
    main()

