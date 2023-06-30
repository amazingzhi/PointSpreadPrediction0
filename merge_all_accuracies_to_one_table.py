import os
import re
import pandas as pd

folder_path = "../accuracies/accuracies_19_test_game/all_before_19"

# create an empty list to hold the data for the dataframe
data = []

# iterate over all the files in the folder
for file_name in os.listdir(folder_path):
    if not file_name.endswith('.txt'):
        continue

    # skip files that don't have pointspread in the name
    if 'pointspread' not in file_name:
        continue
    # determine algorithm name based on file name
    algo_match = re.search(r'(DTR|GBR|Lasso|Light|LR|RF|Ridge|SGD|SVM|DNN)', file_name)
    if algo_match:
        algorithm = algo_match.group(1)
    else:
        algorithm = 'not_need'
    # determine target variable based on file name
    if 'PTS' in file_name:
        target_variable = 'PTS'
    else:
        target_variable = 'pointspread'

    # determine PCA_not based on file name
    if 'PCA' in file_name:
        PCA_not = 'PCA'
    else:
        PCA_not = 'not_PCA'

    # determine transform based on file name
    if 'standard' in file_name:
        transform = 'standard'
    elif 'minmax' in file_name:
        transform = 'minmax'
    else:
        transform = 'not_need'

    # determine home_away based on file name
    if 'home' in file_name:
        home_away = 'home'
    elif 'away' in file_name:
        home_away = 'away'
    else:
        home_away = 'not_need'

    # read the file content and extract the required information
    with open(os.path.join(folder_path, file_name), 'r') as f:
        content = f.read()
        # Find train error
        train_error_match = re.search(r'best_score_:\s*(-?\d+\.\d+)', content)
        if train_error_match:
            train_error = round(float(train_error_match.group(1)), 2)
        else:
            train_error = 'not_need'

        # Find best params
        best_params_match = re.search(r'best_params_:(.*)', content)
        if best_params_match:
            best_params = best_params_match.group(1).strip()
        else:
            best_params = 'not_need'

        # Find rmse
        rmse_match = re.search(r'rooted_mean_squared_error:(\d+\.\d+)', content)
        if rmse_match:
            rmse = round(float(rmse_match.group(1)), 2)
        else:
            rmse = 'not_need'

        # Find max error
        max_error_match = re.search(r'max_error:(\d+\.\d+)', content)
        if max_error_match:
            max_error = round(float(max_error_match.group(1)), 2)
        else:
            max_error = 'not_need'

        # Find mae
        mae_match = re.search(r'mean_absolute_error:(\d+\.\d+)', content)
        if mae_match:
            mae = round(float(mae_match.group(1)), 2)
        else:
            mae = 'not_need'

        # Find r^2
        r2_match = re.search(r'r2_score:(-?\d+\.\d+)', content)
        if r2_match:
            r2 = round(float(r2_match.group(1)), 2)
        else:
            r2 = 'not_need'

    # add the extracted information to the data list
    data.append([target_variable, algorithm, PCA_not, transform, home_away, train_error, best_params, rmse, max_error, mae, r2])

# create the pandas dataframe
df = pd.DataFrame(data, columns=['Target_variable', 'Algorithm', 'PCA_not', 'transform', 'home_away', 'train_error', 'best_params', 'rmse', 'max_error', 'mae', 'r^2'])

# export the dataframe to a csv file
df.to_csv('../Result/master/accuracies_all.csv', index=False)

# Sort the DataFrame
df = df.sort_values(['Algorithm', 'r^2', 'mae', 'rmse', 'max_error'], ascending=[True, False, True, True, True])

# Keep only the best model for each algorithm
df = df.drop_duplicates(subset='Algorithm', keep='first')

# export the dataframe to a csv file
df.to_csv('../Result/master/accuracies_best.csv', index=False)
