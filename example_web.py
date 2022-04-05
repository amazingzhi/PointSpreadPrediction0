## import libraries
import torch
import holidays
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import TensorDataset, DataLoader
# uncomment if you want to create directory checkpoint, best_model!
# mkdir checkpoint best_model!
## see if cuda available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")


## get one hot encoding for pandas dataframe
def onehot_encode_pd(df, cols):
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col)

    return pd.concat([df, dummies], axis=1).drop(columns=cols)


## find which day is holiday in US.
us_holidays = holidays.US()


def is_holiday(date):
    date = date.replace(hour=0)
    return 1 if (date in us_holidays) else 0


def add_holiday_col(df, holidays):
    return df.assign(is_holiday=df.index.to_series().apply(is_holiday))


## get X and y by given pandas df
def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y


def train_test_split1(df, target_col, test_ratio):
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    return X_train, X_test, y_train, y_test


## train validate test split
def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


# X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_features, 'value', 0.2)

## standardization or normalization
def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()


# scaler = get_scaler('minmax')
# X_train_arr = scaler.fit_transform(X_train)
# X_val_arr = scaler.transform(X_val)
# X_test_arr = scaler.transform(X_test)
#
# y_train_arr = scaler.fit_transform(y_train)
# y_val_arr = scaler.transform(y_val)
# y_test_arr = scaler.transform(y_test)

## data from df to tensor and then to dataloader
# batch_size = 64
#
# train_features = torch.Tensor(X_train_arr)
# train_targets = torch.Tensor(y_train_arr)
# val_features = torch.Tensor(X_val_arr)
# val_targets = torch.Tensor(y_val_arr)
# test_features = torch.Tensor(X_test_arr)
# test_targets = torch.Tensor(y_test_arr)
#
# train = TensorDataset(train_features, train_targets)
# val = TensorDataset(val_features, val_targets)
# test = TensorDataset(test_features, test_targets)
#
# train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
# val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
# test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
# test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

### DNN 1 layer
class DNNModel1(nn.Module):
    def __init__(self, input_size, hidden_size1, out_size):
        super(DNNModel1, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.out_size = out_size
        self.leaky_relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.out = nn.Linear(hidden_size1, out_size)

    def forward(self, X):
        out = self.l1(X)
        out = self.leaky_relu(out)
        out = self.out(out)
        # no activation and no softmax at the end
        return out


### DNN 2 layers
class DNNModel2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, out_size):
        super(DNNModel2, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.out_size = out_size
        self.leaky_relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.out = nn.Linear(hidden_size2, out_size)

    def forward(self, X):
        out = self.l1(X)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.out(out)
        # no activation and no softmax at the end
        return out


### DNN 3 layers
class DNNModel3(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, out_size):
        super(DNNModel3, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.out_size = out_size
        self.leaky_relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, hidden_size3)
        self.out = nn.Linear(hidden_size3, out_size)

    def forward(self, X):
        out = self.l1(X)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.l3(out)
        out = self.leaky_relu(out)
        out = self.out(out)
        # no activation and no softmax at the end
        return out


### DNN 4 layers
class DNNModel4(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, out_size):
        super(DNNModel4, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.out_size = out_size
        self.leaky_relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, hidden_size3)
        self.l4 = nn.Linear(hidden_size3, hidden_size4)
        self.out = nn.Linear(hidden_size4, out_size)

    def forward(self, X):
        out = self.l1(X)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.l3(out)
        out = self.leaky_relu(out)
        out = self.l4(out)
        out = self.leaky_relu(out)
        out = self.out(out)
        # no activation and no softmax at the end
        return out


### DNN 5 layers
class DNNModel5(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, out_size):
        super(DNNModel5, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.hidden_size5 = hidden_size5
        self.out_size = out_size
        self.leaky_relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, hidden_size3)
        self.l4 = nn.Linear(hidden_size3, hidden_size4)
        self.l5 = nn.Linear(hidden_size4, hidden_size5)
        self.out = nn.Linear(hidden_size5, out_size)

    def forward(self, X):
        out = self.l1(X)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.l3(out)
        out = self.leaky_relu(out)
        out = self.l4(out)
        out = self.leaky_relu(out)
        out = self.l5(out)
        out = self.leaky_relu(out)
        out = self.out(out)
        # no activation and no softmax at the end
        return out


### DNN 6 layers
class DNNModel6(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6,
                 out_size):
        super(DNNModel6, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.hidden_size5 = hidden_size5
        self.hidden_size6 = hidden_size6
        self.out_size = out_size
        self.leaky_relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, hidden_size3)
        self.l4 = nn.Linear(hidden_size3, hidden_size4)
        self.l5 = nn.Linear(hidden_size4, hidden_size5)
        self.l6 = nn.Linear(hidden_size5, hidden_size6)
        self.out = nn.Linear(hidden_size6, out_size)

    def forward(self, X):
        out = self.l1(X)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.l3(out)
        out = self.leaky_relu(out)
        out = self.l4(out)
        out = self.leaky_relu(out)
        out = self.l5(out)
        out = self.leaky_relu(out)
        out = self.l6(out)
        out = self.leaky_relu(out)
        out = self.out(out)
        # no activation and no softmax at the end
        return out

### DNN 9 layers
class DNNModel9(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6,
                 hidden_size7,hidden_size8,hidden_size9,
                 out_size):
        super(DNNModel9, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.hidden_size5 = hidden_size5
        self.hidden_size6 = hidden_size6
        self.hidden_size7 = hidden_size7
        self.hidden_size8 = hidden_size8
        self.hidden_size9 = hidden_size9
        self.out_size = out_size
        self.leaky_relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, hidden_size3)
        self.l4 = nn.Linear(hidden_size3, hidden_size4)
        self.l5 = nn.Linear(hidden_size4, hidden_size5)
        self.l6 = nn.Linear(hidden_size5, hidden_size6)
        self.l7 = nn.Linear(hidden_size6, hidden_size7)
        self.l8 = nn.Linear(hidden_size7, hidden_size8)
        self.l9 = nn.Linear(hidden_size8, hidden_size9)
        self.out = nn.Linear(hidden_size9, out_size)

    def forward(self, X):
        out = self.l1(X)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.l3(out)
        out = self.leaky_relu(out)
        out = self.l4(out)
        out = self.leaky_relu(out)
        out = self.l5(out)
        out = self.leaky_relu(out)
        out = self.l6(out)
        out = self.leaky_relu(out)
        out = self.l7(out)
        out = self.leaky_relu(out)
        out = self.l8(out)
        out = self.leaky_relu(out)
        out = self.l9(out)
        out = self.leaky_relu(out)
        out = self.out(out)
        # no activation and no softmax at the end
        return out
### RNN model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates an RNN instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out


### LSTM model
class LSTMModel(nn.Module):
    """LSTMModel class extends nn.Module class and works as a constructor for LSTMs.

       LSTMModel class initiates a LSTM module based on PyTorch's nn.Module class.
       It has only two methods, namely init() and forward(). While the init()
       method initiates the model with the given input parameters, the forward()
       method defines how the forward propagation needs to be calculated.
       Since PyTorch automatically defines back propagation, there is no need
       to define back propagation method.

       Attributes:
           hidden_dim (int): The number of nodes in each layer
           layer_dim (str): The number of layers in the network
           lstm (nn.LSTM): The LSTM model constructed with the input parameters.
           fc (nn.Linear): The fully connected layer to convert the final state
                           of LSTMs to our desired output shape.

    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates a LSTM instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out


### GRU
class GRUModel(nn.Module):
    """GRUModel class extends nn.Module class and works as a constructor for GRUs.

       GRUModel class initiates a GRU module based on PyTorch's nn.Module class.
       It has only two methods, namely init() and forward(). While the init()
       method initiates the model with the given input parameters, the forward()
       method defines how the forward propagation needs to be calculated.
       Since PyTorch automatically defines back propagation, there is no need
       to define back propagation method.

       Attributes:
           hidden_dim (int): The number of nodes in each layer
           layer_dim (str): The number of layers in the network
           gru (nn.GRU): The GRU model constructed with the input parameters.
           fc (nn.Linear): The fully connected layer to convert the final state
                           of GRUs to our desired output shape.

    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates a GRU instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out


### choose model to do RNN
def get_model(model, model_params):
    models = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
        "dnn1": DNNModel1,
        "dnn2": DNNModel2,
        "dnn3": DNNModel3,
        "dnn4": DNNModel4,
        "dnn5": DNNModel5,
        "dnn6": DNNModel6,
        "dnn9": DNNModel9
    }
    return models.get(model.lower())(**model_params)


# saving function
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


# loading function
def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


### Optimization is a helper class that allows training, validation, prediction
class Optimization:
    """Optimization is a helper class that allows training, validation, prediction.

    Optimization is a helper class that takes model, loss function, optimizer function
    learning scheduler (optional), early stopping (optional) as inputs. In return, it
    provides a framework to train and validate the models, and to predict future values
    based on the models.

    Attributes:
        model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
        loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
        optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        train_losses (list[float]): The loss values from the training
        val_losses (list[float]): The loss values from the validation
        last_epoch (int): The number of epochs that the models is trained
    """

    def __init__(self, model, loss_fn, optimizer):
        """
        Args:
            model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
            loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
            optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        """The method train_step completes one step of training.

        Given the features (x) and the target values (y) tensors, the method completes
        one step of the training. First, it activates the train mode to enable back prop.
        After generating predicted values (yhat) by doing forward propagation, it calculates
        the losses by using the loss function. Then, it computes the gradients by doing
        back propagation and updates the weights by calling step() function.

        Args:
            x (torch.Tensor): Tensor for features to train one step
            y (torch.Tensor): Tensor for target values to calculate losses

        """
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, checkpoint_path, best_model_path, tb, valid_loss_min_input=np.Inf,
              batch_size=64, n_epochs=50, n_features=1, ):
        """The method train performs the model training

        The method takes DataLoaders for training and validation datasets, batch size for
        mini-batch training, number of epochs to train, and number of features as inputs.
        Then, it carries out the training by iteratively calling the method train_step for
        n_epochs times. If early stopping is enabled, then it  checks the stopping condition
        to decide whether the training needs to halt before n_epochs steps. Finally, it saves
        the model in a designated file path.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader that stores training data
            val_loader (torch.utils.data.DataLoader): DataLoader that stores validation data
            batch_size (int): Batch size for mini-batch training
            n_epochs (int): Number of epochs, i.e., train steps, to train
            n_features (int): Number of feature columns

        """
        # initialize tracker for minimum validation loss
        valid_loss_min = valid_loss_min_input

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                # x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                # y_batch = y_batch.view([batch_size, 1, 1]).to(device)
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    # x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    # y_val = y_val.view([batch_size, 1, 1]).to(device)
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
            tb.add_scalar("Training Loss", training_loss, epoch)
            tb.add_scalar("Validation Loss", validation_loss, epoch)

            if (epoch <= 10) | (epoch % 1 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )
            # create checkpoint variable and add important data
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': validation_loss,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            # save checkpoint
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)

            ## TODO: save the model if validation loss has decreased
            if validation_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                validation_loss))
                # save checkpoint as best model
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                valid_loss_min = validation_loss
        return valid_loss_min

    def evaluate(self, test_loader, model_ori, optimizer_ori,best_model_path,batch_size=1, n_features=1):
        """The method evaluate performs the model evaluation

        The method takes DataLoaders for the test dataset, batch size for mini-batch testing,
        and number of features as inputs. Similar to the model validation, it iteratively
        predicts the target values and calculates losses. Then, it returns two lists that
        hold the predictions and the actual values.

        Note:
            This method assumes that the prediction from the previous step is available at
            the time of the prediction, and only does one-step prediction into the future.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
            batch_size (int): Batch size for mini-batch training
            n_features (int): Number of feature columns

        Returns:
            list[float]: The values predicted by the model
            list[float]: The actual values in the test set.

        """
        # define checkpoint saved path
        ckp_path = best_model_path
        # load the saved checkpoint
        model, optimizer, start_epoch, valid_loss_min = load_ckp(ckp_path, model_ori, optimizer_ori)
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                # x_test = x_test.view([batch_size, -1, n_features]).to(device)
                # y_test = y_test.view([batch_size, 1, 1]).to(device)
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                # self.model.eval()
                # yhat = self.model(x_test)
                model.eval()
                yhat = model(x_test)
                yhat = yhat.view([1])#comment when use for format prediction function in example_web.py
                predictions.append(yhat.to(device).detach().numpy())
                y_test = y_test.view([1])#comment when use for format prediction function in example_web.py
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

    def plot_losses(self):
        """The method plots the calculated loss values for training and validation
        """
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()


# ## Training the model
# import torch.optim as optim
#
# input_dim = len(X_train.columns)
# output_dim = 1
# hidden_dim = 64
# layer_dim = 3
# batch_size = 64
# dropout = 0.2
# n_epochs = 50
# learning_rate = 1e-3
# weight_decay = 1e-6
#
# model_params = {'input_dim': input_dim,
#                 'hidden_dim' : hidden_dim,
#                 'layer_dim' : layer_dim,
#                 'output_dim' : output_dim,
#                 'dropout_prob' : dropout}
#
# model = get_model('lstm', model_params)
#
# loss_fn = nn.MSELoss(reduction="mean")
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#
#
# opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
# opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
# opt.plot_losses()
#
# predictions, values = opt.evaluate(
#     test_loader_one,
#     batch_size=1,
#     n_features=input_dim
# )

## format prediction
def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions, values, df_test, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result


# df_result = format_predictions(predictions, values, X_test, scaler)
# df_result

## Calculating error metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error


def calculate_metrics(df):
    result_metrics = {'mae': mean_absolute_error(df.value, df.prediction),
                      'rmse': mean_squared_error(df.value, df.prediction) ** 0.5,
                      'r2': r2_score(df.value, df.prediction),
                      'max': max_error(df.value, df.prediction)}
    print("RF max error:              ", result_metrics["max"])
    print("Mean Absolute Error:       ", result_metrics["mae"])
    print("Root Mean Squared Error:   ", result_metrics["rmse"])
    print("R^2 Score:                 ", result_metrics["r2"])
    return result_metrics


# result_metrics = calculate_metrics(df_result)

## Generating baseline predictions(linear regression benchmark)
from sklearn.linear_model import LinearRegression


def build_baseline_model(df, test_ratio, target_col):
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    result = pd.DataFrame(y_test)
    result["prediction"] = prediction
    result = result.sort_index()

    return result

# df_baseline = build_baseline_model(df_features, 0.2, 'value')
# baseline_metrics = calculate_metrics(df_baseline)

# ## Visualizing the predictions
# import plotly.offline as pyo
# import plotly.graph_objs as go
# from plotly.offline import iplot
#
#
# def plot_predictions(df_result, df_baseline):
#     data = []
#
#     value = go.Scatter(
#         x=df_result.index,
#         y=df_result.value,
#         mode="lines",
#         name="values",
#         marker=dict(),
#         text=df_result.index,
#         line=dict(color="rgba(0,0,0, 0.3)"),
#     )
#     data.append(value)
#
#     baseline = go.Scatter(
#         x=df_baseline.index,
#         y=df_baseline.prediction,
#         mode="lines",
#         line={"dash": "dot"},
#         name='linear regression',
#         marker=dict(),
#         text=df_baseline.index,
#         opacity=0.8,
#     )
#     data.append(baseline)
#
#     prediction = go.Scatter(
#         x=df_result.index,
#         y=df_result.prediction,
#         mode="lines",
#         line={"dash": "dot"},
#         name='predictions',
#         marker=dict(),
#         text=df_result.index,
#         opacity=0.8,
#     )
#     data.append(prediction)
#
#     layout = dict(
#         title="Predictions vs Actual Values for the dataset",
#         xaxis=dict(title="Time", ticklen=5, zeroline=False),
#         yaxis=dict(title="Value", ticklen=5, zeroline=False),
#     )
#
#     fig = dict(data=data, layout=layout)
#     iplot(fig)
#
#
# # Set notebook mode to work in offline
# pyo.init_notebook_mode()
#
# plot_predictions(df_result, df_baseline)
