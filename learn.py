# Based on:
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py

import torch
import torch.nn as nn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.lin = nn.Linear(256, 1000)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)\
            .to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)\
            .to(device)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        out = self.lin(out)
        return out


def load_dataset(batch_size):
    X = np.concatenate((
        np.load("./resources/param_datapoints_x.npy"),
        np.load("./resources/return_datapoints_x.npy")
    ))

    y = np.concatenate((
        np.argmax(np.load("./resources/param_datapoints_y.npy"), axis=1),
        np.argmax(np.load("./resources/return_datapoints_y.npy"), axis=1)
    ))

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    train_data = torch.utils.data.TensorDataset(X, y)

    train_size = int(0.80 * len(train_data))
    test_size = len(train_data) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def learn():

    sequence_length = 55
    input_size = 100  # The number of expected features in the input `x`
    hidden_size = 128  # 128x2 = 256
    num_layers = 1
    batch_size = 256
    num_epochs = 12
    learning_rate = 0.002

    # Load data
    train_loader, test_loader = load_dataset(batch_size)

    # Load model
    model = BiRNN(input_size, hidden_size, num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss?
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (datapoints, labels) in enumerate(train_loader):
            datapoints = datapoints.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(datapoints)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss:{:.10f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Test the model
    # TODO: this is temporary
    with torch.no_grad():
        correct = 0
        total = 0
        for datapoints, labels in test_loader:
            datapoints = datapoints.to(device)
            labels = labels.to(device)
            outputs = model(datapoints)
            predicted = np.argmax(outputs.data.cpu(), axis=1)
            actual = labels.data.cpu()
            total += labels.size(0)
            correct += (predicted == actual).sum(dim=0)

        print('Test Accuracy of the model: {} %'.format(100.0 * float(correct) / float(total)))


learn()
