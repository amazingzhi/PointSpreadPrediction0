# import libraries
from os import mkdir
from itertools import product
import torch
import torch.nn as nn
import pandas as pd
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import Dataset, TensorDataset, DataLoader, TensorDataset, random_split, SubsetRandomSampler, \
    ConcatDataset
import example_web as ew
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

""" 
ToDo YOU HAVE TO CREATE checkpoint and model folders in your working directory
Todo: after you change datasets, put "tensorboard --logdir runs/ runs_feature_importance/ runs_PCA" into terminal and
    change working directory. after you have run the program, put "tensorboard --logdir runs/ runs_feature_importance/
runs_PCA" into terminal again to see your results in tensorboard. 
todo : change variables with comments after codes when you change datasets"""
# global parameters set up
n_epochs = 99
tb = SummaryWriter()
## see if cuda available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")


def calculate_metrics(y_true, y_pred):
    result_metrics = {'mae': mean_absolute_error(y_true, y_pred),
                      'rmse': mean_squared_error(y_true, y_pred) ** 0.5,
                      'r2': r2_score(y_true, y_pred),
                      'max': max_error(y_true, y_pred)}
    print("RF max error:              ", result_metrics["max"])
    print("Mean Absolute Error:       ", result_metrics["mae"])
    print("Root Mean Squared Error:   ", result_metrics["rmse"])
    print("R^2 Score:                 ", result_metrics["r2"])
    return result_metrics


# read data
Train = pd.read_csv('data/PCA_data/PCA_train.csv')
    # todo choices: 'data/PCA_data/PCA_train.csv'
    # 'data/feature_selection_data/train.csv'
    # 'data/Original_data/train&validate.csv'
Validate = pd.read_csv('data/PCA_data/PCA_validation.csv')
    # todo choices: 'data/Original_data/validation.csv'
    # 'data/PCA_data/PCA_validation.csv'
    # 'data/feature_selection_data/validation.csv'
Test = pd.read_csv('data/PCA_data/PCA_test.csv')
    # todo choices: 'data/PCA_data/PCA_test.csv'
    # 'data/Original_data/test.csv'
    # 'data/feature_selection_data/test_feature_selection.csv'

# Train = Train.iloc[:, 4:]  # todo: comment all these three lines if it is a pca data, but uncomment for other data.
# Validate = Validate.iloc[:, 4:]
# Test = Test.iloc[:, 4:]

# train test split
Train_X, Train_y_ori = ew.feature_label_split(Train, 'pointspread')
Validate_X, Validate_y_ori = ew.feature_label_split(Validate, 'pointspread')
Test_X, Test_y_ori = ew.feature_label_split(Test, 'pointspread')

# normalization
scaler = ew.get_scaler('minmax')
X_train_arr = scaler.fit_transform(Train_X)
X_validate_arr = scaler.fit_transform(Validate_X)
X_test_arr = scaler.transform(Test_X)

y_train_arr = scaler.fit_transform(Train_y_ori)
y_validate_arr = scaler.transform(Validate_y_ori)
y_test_arr = scaler.transform(Test_y_ori)

# data from df to tensor and then to dataloader
train_features = torch.Tensor(X_train_arr)
train_targets = torch.Tensor(y_train_arr)
validate_features = torch.Tensor(X_validate_arr)
validate_targets = torch.Tensor(y_validate_arr)
test_features = torch.Tensor(X_test_arr)
test_targets = torch.Tensor(y_test_arr)

train = TensorDataset(train_features, train_targets)
validate = TensorDataset(validate_features, validate_targets)
test = TensorDataset(test_features, test_targets)

# model parameters
input_dim = len(Train_X.columns)
input_size = input_dim
hidden_size1 = input_dim
hidden_size2 = input_dim
hidden_size3 = input_dim
hidden_size4 = (input_dim // 3) * 2
hidden_size5 = (input_dim // 3) * 2
hidden_size6 = (input_dim // 3) * 2
hidden_size7 = input_dim // 3
hidden_size8 = input_dim // 3
hidden_size9 = input_dim // 3
out_size = 1

model_params = {
    'model_params1': {'input_size': input_size, 'hidden_size1': hidden_size1, 'out_size': out_size},
    'model_params2': {'input_size': input_size, 'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2,
                      'out_size': out_size},
    'model_params3': {'input_size': input_size, 'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2,
                      'hidden_size3': hidden_size3,
                      'out_size': out_size},
    'model_params4': {'input_size': input_size, 'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2,
                      'hidden_size3': hidden_size3,
                      'hidden_size4': hidden_size4, 'out_size': out_size},
    'model_params5': {'input_size': input_size, 'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2,
                      'hidden_size3': hidden_size3,
                      'hidden_size4': hidden_size4, 'hidden_size5': hidden_size5, 'out_size': out_size},
    'model_params6': {'input_size': input_size, 'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2,
                      'hidden_size3': hidden_size3,
                      'hidden_size4': hidden_size4, 'hidden_size5': hidden_size5, 'hidden_size6': hidden_size6,
                      'out_size': out_size},
    'model_params9': {'input_size': input_size, 'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2,
                      'hidden_size3': hidden_size3,
                      'hidden_size4': hidden_size4, 'hidden_size5': hidden_size5, 'hidden_size6': hidden_size6,
                      'hidden_size7': hidden_size7, 'hidden_size8': hidden_size8, 'hidden_size9': hidden_size9,
                      'out_size': out_size}
}

# training parameters
training_parameters = dict(
    learning_rate=[1e-4, 1e-3, 1e-2],
    batch_size=[64, 32, 16],
    weight_decay=[1e-6, 1e-4, 1e-2],
    model_params=list(model_params.keys())
)
param_values = [v for v in training_parameters.values()]

# training with cross validation
validate_loss_min = np.inf
for run_id, (lr, batch_size, weight_decay, model_param) in enumerate(product(*param_values)):
    print("parameter set id:", run_id + 1)
    # add tensorboard
    print(f'training parameters: learning rate-{lr} batch_size-{batch_size}, weight decay-{weight_decay}, model ' \
          f'parameters-{model_param}')
    comment = f'learning_rate={lr} batch_size={batch_size} weight_decay={weight_decay} model_parameters={model_param}'
    tb = SummaryWriter(log_dir=f'runs_PCA/learning_rate={lr} batch_size={batch_size} weight_decay={weight_decay} model_parameters={model_param}', comment=comment)  # Todo: after you change datasets, put
    # "tensorboard --logdir runs/ runs_feature_importance/ runs_PCA"
    train_loader = DataLoader(train, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(validate, batch_size=batch_size, drop_last=True)
    model = ew.get_model(f'dnn{model_param[-1]}', model_params[model_param])
    X, y = next(iter(train_loader))
    tb.add_graph(model, X)
    loss_fn = nn.L1Loss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    opt = ew.Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
    valid_loss_by_given_params = opt.train(train_loader, val_loader,
                                           "./checkpoint/current_checkpoint.pt",
                                           f"./models/DNN_PCA_{lr}_{batch_size}_{weight_decay}_{model_param}.pt",
                                           # ToDo:
                                           # f"./models/DNN_{lr}_{batch_size}_{weight_decay}_{model_param}.pt"
                                           # f"./models/DNN_feature_importance_{lr}_{batch_size}_{weight_decay}_{model_param}.pt"
                                           # f"./models/DNN_PCA_{lr}_{batch_size}_{weight_decay}_{model_param}.pt"
                                           tb = tb,
                                           batch_size=batch_size, n_epochs=n_epochs,
                                           n_features=input_dim)
    if valid_loss_by_given_params < validate_loss_min:
        print('There is a better set of parameters.')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(validate_loss_min,
                                                                                        valid_loss_by_given_params))
        best_model_path = f"./models/DNN_PCA_{lr}_{batch_size}_{weight_decay}_{model_param}.pt"  # todo
        validate_loss_min = valid_loss_by_given_params
        best_parameters = f'learning_rate={lr} batch_size={batch_size} weight_decay={weight_decay} model_parameters={model_param}'
        best_model = model
        best_optimizer = optimizer
        best_opt = opt
    print("__________________________________________________________")

    tb.add_hparams(
        {"lr": lr, "bsize": batch_size, "weight_decay": weight_decay, "layers": model_param},
        {
            "loss": valid_loss_by_given_params,
        },
    )

tb.close()
print(best_parameters + ';   min loss: ' + str(validate_loss_min))

# load test dataset
test_loader = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

# use best model to predict using test dataset.
predictions, values = best_opt.evaluate(
    test_loader,
    model_ori=best_model,
    optimizer_ori=best_optimizer,
    best_model_path=best_model_path,
    batch_size=1,
    n_features=input_dim
)
predictions = scaler.inverse_transform(predictions)
values = scaler.inverse_transform(values)
result_matrix = calculate_metrics(values, predictions)

# check accuracy measures
result_matrix

# save predictions to csv files
resultCSVPath = r'predictions/PCA_DNN.csv'  # todo: 'predictions/PCA_DNN.csv'
    # 'predictions/feature_selection_DNN.csv'
    # 'predictions/DNN.csv'
pd.DataFrame(predictions).to_csv(resultCSVPath, index=False, na_rep=0)
