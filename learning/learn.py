import os
import pickle
import time
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils import data
from torch.utils.data import DataLoader
from typing import Tuple

import config

import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_model_parameters(model: nn.Module) -> int:
    """
    Count the amount of parameters of a model
    :param model:
    :return:
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def store_model(model: nn.Module, filename: str, model_dir=config.MODEL_DIR) -> None:
    """
    Store the model to a pickle file
    :param model: the model itself
    :param filename: name of the file
    :param model_dir: directory in which to write to file to
    :return:
    """
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, filename), 'wb') as f:
        pickle.dump(model, f)


def store_json(model: nn.Module, filename: str, model_dir=config.MODEL_DIR) -> None:
    """
    Store the model as a json file.
    :param model: the model itself
    :param filename: name of the file
    :param model_dir: directory in which to write to file to
    :return:
    """
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, filename), 'w') as f:
        f.write(json.dumps(model))


def load_model(filename: str, model_dir=config.MODEL_DIR) -> nn.Module:
    """
    Load the model from a pickle.
    :param filename: name of the file
    :param model_dir: directory in which the file is located
    :return:
    """
    with open(os.path.join(model_dir, filename), 'rb') as f:
        return pickle.load(f)


class BiRNN(nn.Module):
    """
    The BiRNN represents the implementation of the Bidirectional RNN model
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional=True) -> None:
        super(BiRNN, self).__init__()

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.dropout = nn.Dropout(p=0.2)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=bidirectional)

        self.linear = nn.Linear(hidden_size * (2 if bidirectional else 1), 1000)

    def forward(self, x):
        x = self.dropout(x)

        # Forward propagate LSTM
        # Out: tensor of shape (batch_size, seq_length, hidden_size*2)
        x, _ = self.lstm(x)

        # Decode the hidden state of the last time step
        x = x[:, -1, :]

        # Output layer
        x = self.linear(x)

        return x


class GRURNN(nn.Module):
    """
    The GRURNN represents the implementation of the GRU RNN model
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional=True) -> None:
        super(GRURNN, self).__init__()

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.dropout = nn.Dropout(p=0.2)

        self.lstm = nn.GRU(input_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=bidirectional)

        self.linear = nn.Linear(hidden_size * (2 if bidirectional else 1), 1000)

    def forward(self, x):
        x = self.dropout(x)

        # Forward propagate LSTM
        # Out: tensor of shape (batch_size, seq_length, hidden_size*2)
        x, _ = self.lstm(x)

        # Decode the hidden state of the last time step
        x = x[:, -1, :]

        # Output layer
        x = self.linear(x)

        return x


def load_dataset(X, y, batch_size: int, split=0.8) -> Tuple:
    """
    Load and return a specific dataset
    :param X: x input part of the dataset
    :param y: y input part of the dataset
    :param batch_size: size to use for the batching
    :param split: amount of data to split (between 0 and 1)
    :return: tuple consisting out of a training and test set
    """
    train_data = torch.utils.data.TensorDataset(X, y)

    train_size = int(split * len(train_data))
    test_size = len(train_data) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(train_data, [train_size, test_size], torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def load_data_tensors(filename_X: str, filename_y: str, limit: int) -> Tuple:
    """
    Load the tensor dataset
    :param filename_X: x input part of the dataset
    :param filename_y: y input part of the dataset
    :param limit: max amount of y data to load in
    :return: Tuple (X,y) consisting out of the tensor dataset
    """
    X = torch.from_numpy(np.load(filename_X)).float()
    y_load = np.load(filename_y)
    y = torch.from_numpy(np.argmax(y_load, axis=1)).long()
    return X, y


def make_batch_prediction(model: nn.Module, X, top_n=1):
    model.eval()
    with torch.no_grad():
        # Compute model output
        outputs = model(X)
        # Max for each label
        labels = np.argsort(outputs.data.cpu().numpy(), axis=1)
        labels = np.flip(labels, axis=1)
        labels = labels[:, :top_n]
        return outputs, labels


def evaluate(model: nn.Module, data_loader: DataLoader, top_n=1):
    predicted_labels = []

    for i, (batch, labels) in enumerate(data_loader):
        _, batch_labels = make_batch_prediction(model, batch.to(device), top_n=top_n)
        predicted_labels.append(batch_labels)

    #print(true_labels)
    predicted_labels = np.vstack(predicted_labels)

    return predicted_labels


def top_n_fix(y_true, y_pred, n):
    best_predicted = np.empty_like(y_true)
    for i in range(y_true.shape[0]):
        if y_true[i] in y_pred[i, :n]:
            best_predicted[i] = y_true[i]
        else:
            best_predicted[i] = y_pred[i, 0]

    return best_predicted


def train_loop(model: nn.Module, data_loader: DataLoader, model_config: dict, model_store_dir, save_each_x_epochs=25):
    model.train()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])
    losses = []

    # Train the model
    total_step = len(data_loader)
    losses = np.empty(total_step * model_config['num_epochs'])
    i = 0
    for epoch in range(1, model_config['num_epochs'] + 1):
        for batch_i, (batch, labels) in enumerate(data_loader):
            batch = batch.to(device)
            labels = labels.to(device)
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            losses[i] = loss.item()
            i += 1

        print(f'Epoch [{epoch}/{model_config["num_epochs"]}], Batch: [{batch_i}/{total_step}], '
              f'Loss:{loss.item():.10f}')
        if device == 'cuda':
            print(f"Cuda v-memory allocated {torch.cuda.memory_allocated()}")

        if epoch % save_each_x_epochs == 0 or (epoch == model_config['num_epochs']):
            print("Storing model!")
            store_model(model, f"model_{model.__class__.__name__}_e_{epoch}_l_{loss.item():0.10f}.h5",
                        model_dir=os.path.join(config.MODEL_DIR, model_store_dir))

    return losses


def load_m1():
    model_config = {
        'sequence_length': 55,
        'input_size': 14,  # The number of expected features in the input `x`
        'hidden_size': 14,  # 128x2: 256
        'num_layers': 2,
        'batch_size': 256,
        'num_epochs': 100,
        'learning_rate': 0.002,
        'bidirectional': True
    }
    # Load the model
    model = BiRNN(model_config['input_size'], model_config['hidden_size'],
                  model_config['num_layers'], model_config['bidirectional']).to(device)
    return model, model_config


def load_m2():
    model_config = {
        'sequence_length': 55,
        'input_size': 14,  # The number of expected features in the input `x`
        'hidden_size': 10,  # 128x2: 256
        'num_layers': 1,
        'batch_size': 256,
        'num_epochs': 100,
        'learning_rate': 0.002,
        'bidirectional': False
    }
    # Load the model
    model = GRURNN(model_config['input_size'], model_config['hidden_size'],
                   model_config['num_layers'], model_config['bidirectional']).to(device)
    return model, model_config


def load_m3():
    model_config = {
        'sequence_length': 55,
        'input_size': 14,  # The number of expected features in the input `x`
        'hidden_size': 128,  # 128x2: 256
        'num_layers': 1,
        'batch_size': 256,
        'num_epochs': 25,
        'learning_rate': 0.002,
        'bidirectional': True
    }
    # Load the model
    model = BiRNN(model_config['input_size'], model_config['hidden_size'],
                  model_config['num_layers'], model_config['bidirectional']).to(device)
    return model, model_config


def load_m4():
    model_config = {
        'sequence_length': 55,
        'input_size': 14,  # The number of expected features in the input `x`
        'hidden_size': 20,  # 128x2: 256
        'num_layers': 1,
        'batch_size': 256,
        'num_epochs': 100,
        'learning_rate': 0.002,
        'bidirectional': True
    }
    # Load the model
    model = BiRNN(model_config['input_size'], model_config['hidden_size'],
                  model_config['num_layers'], model_config['bidirectional']).to(device)
    return model, model_config


def get_datapoints(dataset: str = "") -> Tuple[str, str, str, str, str, str]:
    base = f"./output/vectors/"
    return base + "return_datapoints_x.npy", \
           base + "return_datapoints_y.npy", \
           base + "param_datapoints_x.npy", \
           base + "param_datapoints_y.npy", \
           base + "param_datapoints_y_src.npy", \
           base + "return_datapoints_y_src.npy"


def report(y_true, y_pred, top_n, filename: str):
    # Fix the predictions if the true value is in top-n predictions
    y_pred_fixed = top_n_fix(y_true, y_pred, top_n)

    # Computation of metrics
    report = classification_report(y_true, y_pred_fixed, output_dict=True)
    store_model(report, f"{filename}.pkl", "./output/reports/pkl")
    store_json(report, f"{filename}.json", "./output/reports/json")


def report_loss(losses, filename: str):
    store_model(losses, f"{filename}.pkl", "./output/reports/pkl")
    store_json({"loss": list(losses)}, f"{filename}.json", "./output/reports/json")


def get_enc_to_type_mapping():
    df = pd.read_csv('./output/ml_inputs/_most_frequent_types.csv', index_col=0)
    types_dict = dict()

    def add_to_dict(row: pd.Series):
        types_dict[row.enc] = row.type

    df.apply(lambda row: add_to_dict(row), axis=1)

    return types_dict


if __name__ == '__main__':
    print(f"-- Using {device} for training.")

    top_n_pred = [1, 2, 3]
    models = [load_m3]
    dataset = "dataset"
    n_repetitions = 1

    RETURN_DATAPOINTS_X, RETURN_DATAPOINTS_Y, PARAM_DATAPOINTS_X, PARAM_DATAPOINTS_Y, SRC_PARAM, SRC_RETURN \
        = get_datapoints(dataset)
    print(f"-- Loading data: {dataset}")
    Xr, yr = load_data_tensors(RETURN_DATAPOINTS_X, RETURN_DATAPOINTS_Y, limit=-1)
    Xp, yp = load_data_tensors(PARAM_DATAPOINTS_X, PARAM_DATAPOINTS_Y, limit=-1)
    X = torch.cat((Xp, Xr))
    y = torch.cat((yp, yr))

    yp_src = np.load(SRC_PARAM, allow_pickle=True)
    yr_src = np.load(SRC_RETURN, allow_pickle=True)

    y_src = np.vstack((yp_src, yr_src))

    types_dict = get_enc_to_type_mapping()

    model, model_config = load_m3()

    print(f"-- Model Loaded: {model} with {count_model_parameters(model)} parameters.")

    train_loader, test_loader = load_dataset(X, y, model_config['batch_size'], split=0.8)

    n = len(X)
    train_size = int(0.8 * n)
    test_size = n - train_size
    print(train_size, test_size)
    _, test_idx = torch.utils.data.random_split(range(n), [train_size, test_size], torch.Generator().manual_seed(42))
    y_src_test = y_src[test_idx.indices]

    # Start training
    losses = train_loop(model, train_loader, model_config,
                        model_store_dir=f"{load_model.__name__}/{dataset}" + str(int(time.time())))

    # print("-- Loading model")

    # Evaluate model performance
    y_true, y_pred = evaluate(model, test_loader, top_n=max(top_n_pred))

    # If the prediction is "other" - ignore the result
    idx_of_other = pickle.load(open(f'./output/ml_inputs/label_encoder.pkl', 'rb')).transform(['other'])[0]
    idx = (y_true != idx_of_other) & (y_pred[:, 0] != idx_of_other)

    types_true = np.array([[types_dict[type] for type in y_true[idx]]])
    types_pred = np.array([[types_dict[type[0]] for type in y_pred[idx]]])

    true_types_src = np.concatenate((y_src_test[idx], types_true.T), axis=1)
    pred_types_src = np.concatenate((y_src_test[idx], types_pred.T), axis=1)

    with open("./output/reports/types/true.csv", "w") as f:
        ans = "file;lineno;name;type;element\n"
        for types in true_types_src:
            ans += f'{types[0]};{types[1]};{types[2]};{types[4]};{types[3]}\n'
        f.write(ans)

    with open("./output/reports/types/predicted.csv", "w") as f:
        ans = "file;lineno;name;type;element\n"
        for types in pred_types_src:
            ans += f'{types[0]};{types[1]};{types[2]};{types[4]};{types[3]}\n'
        f.write(ans)

