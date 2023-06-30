import pandas as pd
import example_web as ew
import csv

# global variables
directory = '../predictions/predictions_19_test_game/all_before_19/best_predictions_each_algorithm'
bet_path = '../data/final/betting spread/cleaned_betting_line_2019.csv'
dow_path = '../data/final/stock_index/Dow Jones Industrial Average Historical Data.csv'
sp500_path = '../data/final/stock_index/sp500_index.csv'
BTC_path = '../data/final/stock_index/BTC_cleaned.csv'
commodity_path = '../data/final/stock_index/DowJonesCommodity.csv'
year = '2019'
games_all_simple = 1259
games_all_kelly = 1259
games_one = 80
algorithms = ['DNN', 'SGD', 'LR', 'RF', 'SVM', 'DTR', 'Light', 'GBR', 'Lasso', 'Ridge']
new_columns_TF = [alg + '_TF' for alg in algorithms]
new_columns_AS = [alg + '_AS' for alg in algorithms] + ['home_AS', 'away_AS', 'favorite_AS']
plot_mapping = {'DNN': 'Artificial_Neural_Networks', 'SGD': 'Stochastic_Gradient_Descent', 'LR': 'Ordinary_Least_Squared',
                'RF': 'Random_Forest', 'SVM': 'Support_Vector_Machine', 'DTR': 'Decision_Tree_Regression',
                'Light': 'Light_Gradient_Boosting_Machine', 'GBR': 'Extreme_Gradient_Boosting',
                'Lasso': 'Lasso', 'Ridge': 'Ridge'}
b_pointspread = 0.95
initial_stake = 100
common_columns = ["GAME_ID", "GAME_DATE_EST", "team_id", "oppo_id", "PTS_team", "pointspread", "loc"]
map_home_away = {'DNN': 'away', 'LR': 'home', 'RF': 'away', 'SGD': 'home', 'SVM': 'away', 'Light': 'home',
                 'Ridge': 'away', 'Lasso': 'home', 'GBR': 'home', 'DTR': 'home'}

def main():
    # create profit simulation object
    S = ew.ProfitSimulation(directory, bet_path, dow_path, sp500_path, BTC_path, commodity_path,
                            games_all_simple, games_all_kelly, games_one, new_columns_TF, new_columns_AS, b_pointspread,
                            initial_stake, year, common_columns)
    # read all stocks data.
    btc, com, dow, sp = S.read_indexes_data()
    # clean all stocks data.
        # clean dow data to make it same pattern with other 3 stock datasets.
    dow = S.dow_data_clean(dow)
        # clean all stocks data: calculate returns of each day in 2019 and keep only dates and returns.
    stocks = S.stocks_data_clean(btc, com, dow, sp)
        # merge all stocks data
    merged_stocks = S.stocks_data_merge(stocks)
    # open all predictions files of test data from best_predictions folder and save them in a Pandas DataFrame.
    df = S.merge_all_predictions_to_a_DF()
    # clean DNN column
    df['DNN'] = df['DNN'].str.strip('[]')
    df['DNN'] = df['DNN'].astype(float)
    # remove half data because overlapping
    away_keys = [key for key, value in map_home_away.items() if value == 'away']
    df1 = df[df['loc'] == 1].drop(columns=away_keys)
    df2 = df[df['loc'] == 0][['GAME_ID'] + away_keys]
    merged_df = pd.merge(df1, df2, on='GAME_ID')
    merged_df[away_keys] = merged_df[away_keys].apply(lambda x: x * -1)
    merged_df = merged_df.set_index('GAME_ID')
    merged_df.index = merged_df.index.astype(str)
    # open betting.csv and save it as a Pandas DataFrame
    betting = S.import_betting_data_and_clean()
    # add correct bets columns and point spread column
    betting = S.add_correct_bet_columns(betting)
    # merge betting and predictions to a new dataframe by their game ids. Lost some observations from test dataset
    # because missing values in betting data.
    df = S.merge_betting_and_prediction_by_gameid(merged_df, betting)
    # plot prediction comparison
    # S.prediction_comparision_plot(df)
    # add always bet home team algorithm as an traditional betting method.
    df = S.add_always_bet_home_away(df, S.games_all_simple)
    # add always bet favorite team algorithm as an traditional betting method.
    df = S.add_always_bet_favorite(df, S.games_all_simple)
    # add columns to df that whether point spread prediction is True or False for each algorithm.
    df = S.add_t_or_f_algorithms(df, S.games_all_simple, 'pointspread')
    # add columns to df that whether game winner prediction is True or False for each algorithm.
    df = S.add_t_or_f_algorithms(df, S.games_all_simple, 'gamewinner')
    # add columns to df that accounts for accumulative sum of the profits for each algorithm for all games by the
    #  simple betting rule.
    df = S.add_accumulative_sum_to_df(df, True, 'pointspread', False)
    df = S.add_accumulative_sum_to_df(df, True, 'gamewinner', False)
    # plot accumulative sum of profits for all games by simple betting rule
    S.plot_accumulative_sum(df, S.games_all_simple, 0, False, algorithms, plot_mapping)
    # add columns to df that accounts for accumulative sum of the profits for each algorithm for all games by the Kelly criterion.
    df = S.add_accumulative_sum_to_df(df, True, 'pointspread', True)
    df = S.add_accumulative_sum_to_df(df, True, 'gamewinner', True)
    # df to csv to save accumulative profits by kelly.
    # resultCSVPath = '../results/accumulate_sum_profits_kelly.csv'
    # df.to_csv(resultCSVPath, index=False, na_rep=0)
    # plot accumulative sum of profits for all games in test dataset.
    S.plot_accumulative_sum(df, S.games_all_kelly, 100, True, algorithms, plot_mapping)
    # build a new pandas dataframe to select the most deviate betting line each day as the best bet and save all days best bet into
    # this dataframe for both pointspread and gamewinner bettings.
    df_best = S.create_df_records_best_bet(df, 'pointspread')
    # reverse dataframe to sort date.
    df_best['date'] = pd.to_datetime(df_best['date'], format='%m/%d/%Y')
    df_best = df_best.sort_values(by='date', ascending=True)
    df_best = df_best.set_index([pd.Index(list(range(df_best.index.size)))])
    # add columns to df that accounts for accumulative sum of the returns for each algorithm for one game per day in
    # test dataset.
    df_best = S.add_accumulative_sum_to_df(df_best, False, 'pointspread', True)
    # add columns to df that accounts for returns
    df_best = S.accumulative_sum_to_returns(df_best, algorithms)
    df_best['date'] = df_best['date'].dt.date
    df_best = df_best.sort_values('date')
    # add stock indexes and commodity and bitcoin returns on the same time period.
    # df_best = merged_stocks.merge(df_best, how='left', on='date')
    # df_best.fillna(method='ffill', inplace=True)
    S.plot_accumulative_profit_master_paper(df=df_best, fin_assets=False, sports_assets=True, log=False, algorithms=algorithms,
                               plot_mapping=plot_mapping, year=year)
    # plot accumulative sum of returns where x-axis is the date in 2019.
    S.plot_accumulative_profit_master_paper(df=df_best, fin_assets=True, sports_assets=False, log=False, algorithms=algorithms,
                               plot_mapping=plot_mapping, year=year)
    S.plot_accumulative_profit_master_paper(df=df_best, fin_assets=False, sports_assets=True, log=False, algorithms=algorithms,
                               plot_mapping=plot_mapping, year=year)
    S.plot_accumulative_profit_master_paper(df=df_best, fin_assets=True, sports_assets=True, log=False, algorithms=algorithms,
                               plot_mapping=plot_mapping, year=year)
    #S.plot_accumulative_return(df=df_best, fin_assets=True, sports_assets=True, log=False)
    print('finished')


if __name__ == "__main__":
    main()
