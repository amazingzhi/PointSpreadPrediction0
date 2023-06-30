import pandas as pd
import example_web as ew
df = pd.read_csv('../predictions/predictions_19_test_game/all_before_19/best_predictions_each_algorithm/pred_each_alg_two_same_gameids.csv')
for column in df.iloc[:, 7:]:
    print(column + ' : accuracy:')
    ew.print_accuracy_measures(df['pointspread'],df[column])
