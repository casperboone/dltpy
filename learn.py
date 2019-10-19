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

MODEL_DIR = "./output/models/"
RETURN_DATAPOINTS_X = "./output/vectors/return_datapoints_x.npy"
RETURN_DATAPOINTS_Y = "./output/vectors/return_datapoints_y.npy"
PARAM_DATAPOINTS_X = "./output/vectors/param_datapoints_x.npy"
PARAM_DATAPOINTS_Y = "./output/vectors/param_datapoints_y.npy"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def store_model(model, filename, model_dir=MODEL_DIR):
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, filename), 'wb') as f:
        pickle.dump(model, f)


def store_json(model, filename, model_dir=MODEL_DIR):
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, filename), 'w') as f:
        f.write(json.dumps(model))


def load_model(filename, model_dir=MODEL_DIR):
    with open(os.path.join(model_dir, filename), 'rb') as f:
        return pickle.load(f)


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True):
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
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True):
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


def load_dataset(X, y, batch_size, split=0.8):
    train_data = torch.utils.data.TensorDataset(X, y)

    train_size = int(split * len(train_data))
    test_size = len(train_data) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def load_data_tensors(filename_X, filename_y, limit):
    X = torch.from_numpy(np.load(filename_X)[0:limit]).float()
    y = torch.from_numpy(np.argmax(np.load(filename_y), axis=1)[0:limit]).long()
    return X, y


def make_batch_prediction(model, X, top_n=1):
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
    true_labels = []
    predicted_labels = []

    for i, (batch, labels) in enumerate(data_loader):
        _, batch_labels = make_batch_prediction(model, batch.to(device), top_n=top_n)
        predicted_labels.append(batch_labels)
        true_labels.append(labels)

    true_labels = np.hstack(true_labels)
    predicted_labels = np.vstack(predicted_labels)

    return true_labels, predicted_labels

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

    # Train the model
    total_step = len(data_loader)

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

            if batch_i % 100 == 0:
                print(f'Epoch [{epoch}/{model_config["num_epochs"]}], Batch: [{batch_i}/{total_step}], '
                      f'Loss:{loss.item():.10f}')
                if device == 'cuda':
                    print(f"Cuda v-memory allocated {torch.cuda.memory_allocated()}")

        if epoch % save_each_x_epochs == 0 or (epoch == model_config['num_epochs'] + 1):
            print("Storing model!")
            store_model(model, f"model_{model.__class__.__name__}_e_{epoch}_l_{loss.item():0.10f}.h5",
                        model_dir=os.path.join(MODEL_DIR, model_store_dir))


def load_m1():
    model_config = {
        'sequence_length': 55,
        'input_size': 14,  # The number of expected features in the input `x`
        'hidden_size': 14,  # 128x2: 256
        'num_layers': 2,
        'batch_size': 128,
        'num_epochs': 500,
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
        'batch_size': 128,
        'num_epochs': 500,
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


def get_datapoints(dataset: str):
    base = f"./input_datasets/{dataset}/vectors/"
    return base + "return_datapoints_x.npy", base + "return_datapoints_y.npy", base + "param_datapoints_x.npy", base + "param_datapoints_y.npy"



def report(y_true, y_pred, top_n, filename):
    # Fix the predictions if the true value is in top-n predictions
    y_pred_fixed = top_n_fix(y_true, y_pred, top_n)

    # Computation of metrics
    report = classification_report(y_true, y_pred_fixed, output_dict=True)
    store_model(report, f"{filename}.pkl", "./output/reports/")
    store_json(report, f"{filename}.json", "./output/reports/")


if __name__ == '__main__':
    print(f"-- Using {device} for training.")

    top_n_pred = [1,2,3]
    models = [load_m1, load_m2, load_m3]
    datasets = ["1_complete", "2_cf_cr_optional", "3_cp_cf_cr_optional", "4_complete_without_return_expressions"]
    n_repetitions = 3

    for dataset in datasets:
        # Load data
        RETURN_DATAPOINTS_X, RETURN_DATAPOINTS_Y, PARAM_DATAPOINTS_X, PARAM_DATAPOINTS_Y = get_datapoints(dataset)
        print(f"-- Loading data: {dataset}")
        Xr, yr = load_data_tensors(RETURN_DATAPOINTS_X, RETURN_DATAPOINTS_Y, limit=-1)
        Xp, yp = load_data_tensors(PARAM_DATAPOINTS_X, PARAM_DATAPOINTS_Y, limit=-1)
        X = torch.cat((Xp, Xr))
        y = torch.cat((yp, yr))

        for load_model in models:
            for i in range(n_repetitions):
                model, model_config = load_model()

                print(f"-- Model Loaded: {model} with {count_model_parameters(model)} parameters.")

                train_loader, test_loader = load_dataset(X, y, model_config['batch_size'], split=0.8)

                # Start training
                train_loop(model, train_loader, model_config, model_store_dir=str(int(time.time())))

                # print("-- Loading model")
                # model = load_model('1571306801/model_BiRNN_e_9_l_1.8179169893.h5')

                # Evaluate model performance
                y_true, y_pred = evaluate(model, test_loader, top_n=max(top_n_pred))

                # If the prediction is "other" - ignore the result
                idx_of_other = pickle.load(open(f'./input_datasets/{dataset}/ml_inputs/label_encoder.pkl', 'rb')).transform(['other'])[0]
                idx = (y_true != idx_of_other) & (y_pred[:, 0] != idx_of_other)

                for top_n in top_n_pred:
                    filename = f"{type(model).__name__}_{dataset}_{i}_{top_n}"
                    report(y_true, y_pred, top_n, filename)
                    report(y_true[idx], y_pred[idx], top_n, f"{filename}_unfiltered")
