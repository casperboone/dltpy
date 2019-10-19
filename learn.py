import os
import pickle
import time

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
        super(BiRNN, self).__init__()

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


def load_dataset(filename_X, filename_y, batch_size, limit=None, split=0.8):
    X = torch.from_numpy(np.load(filename_X)[0:limit]).float()
    y = torch.from_numpy(np.argmax(np.load(filename_y), axis=1)[0:limit]).long()

    train_data = torch.utils.data.TensorDataset(X, y)

    train_size = int(split * len(train_data))
    test_size = len(train_data) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def make_batch_prediction(model, X):
    model.eval()
    with torch.no_grad():
        # Compute model output
        outputs = model(X)
        # Max for each label
        labels = torch.argmax(outputs, 1)
        return outputs, labels


def evaluate(model: nn.Module, data_loader: DataLoader):
    true_labels = []
    predicted_labels = []

    for i, (batch, labels) in enumerate(data_loader):
        _, batch_labels = make_batch_prediction(model, batch.to(device))
        predicted_labels.append(batch_labels)
        true_labels.append(labels)

    true_labels = np.hstack(true_labels)
    predicted_labels = np.hstack(predicted_labels)

    return true_labels, predicted_labels


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
        'hidden_size': 10,  # 128x2: 256
        'num_layers': 1,
        'batch_size': 32,
        'num_epochs': 500,
        'learning_rate': 0.002,
        'bidirectional': False
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
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.002,
        'bidirectional': False
    }
    # Load the model
    model = GRURNN(model_config['input_size'], model_config['hidden_size'],
                  model_config['num_layers'], model_config['bidirectional']).to(device)
    return model, model_config


if __name__ == '__main__':
    print(f"-- Using {device} for training.")
    model, model_config = load_m1()

    print(f"-- Model Loaded: {model} with {count_model_parameters(model)} parameters.")

    # Load data
    print("-- Loading data")
    train_loader, test_loader = load_dataset(RETURN_DATAPOINTS_X, RETURN_DATAPOINTS_Y, model_config['batch_size'],
                                             limit=-1, split=0.8)

    # Start training
    train_loop(model, train_loader, model_config, model_store_dir=str(int(time.time())))

    # print("-- Loading model")
    # model = load_model('1571306801/model_BiRNN_e_9_l_1.8179169893.h5')

    # Evaluate model performance
    y_true, y_pred = evaluate(model, test_loader)

    # Computation of metrics...... more to come?
    print(classification_report(y_true, y_pred))
