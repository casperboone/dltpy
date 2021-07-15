import pickle
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from torch.nn.functional import pad
from torch.utils import data

# Device configuration
from learning.learn import get_datapoints, load_data_tensors, BiRNN, load_m3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_ort_prediction(ort_session, X, model_config, top_n):
    h0 = torch.zeros((2 if model_config['bidirectional'] else 1) * model_config['num_layers'],
                     model_config['batch_size'], model_config['hidden_size']).to(device)
    c0 = torch.zeros((2 if model_config['bidirectional'] else 1) * model_config['num_layers'],
                     model_config['batch_size'], model_config['hidden_size']).to(device)
    out_size = X.shape[0]
    if X.shape[0] < model_config['batch_size']:
        X = pad(X, (0, 0, 0, 0, 0, model_config['batch_size'] - X.shape[0]), 'constant', 0)

    y_onnx = ort_session.run(None,
                             {'input': X.cpu().numpy(), 'h0': h0.cpu().numpy(), 'c0': c0.cpu().numpy()})

    # print(np.array(y_onnx).shape)

    # Max for each label
    labels = np.argsort(np.array(y_onnx[0]), axis=1)
    labels = np.flip(labels, axis=1)
    labels = labels[:, :top_n]
    return y_onnx[0][:out_size], labels[:out_size]


def get_enc_to_type_mapping():
    df = pd.read_csv('../output/ml_inputs/_most_frequent_types.csv', index_col=0)
    types_dict = dict()

    def add_to_dict(row: pd.Series):
        types_dict[row.enc] = row.type

    df.apply(lambda row: add_to_dict(row), axis=1)

    return types_dict


def evaluate_onnx(ort_session, data_loader, model_config, top_n):
    # true_labels = []
    predicted_labels = []

    for i, (batch, labels) in enumerate(data_loader):
        _, batch_labels = make_ort_prediction(ort_session, batch.to(device), model_config, top_n=top_n)
        predicted_labels.append(batch_labels)
        # true_labels.append(labels)

    # true_labels = np.hstack(true_labels)
    # print(true_labels)
    predicted_labels = np.vstack(predicted_labels)

    return predicted_labels


if __name__ == '__main__':
    t1 = time.time_ns()
    RETURN_DATAPOINTS_X, RETURN_DATAPOINTS_Y, PARAM_DATAPOINTS_X, PARAM_DATAPOINTS_Y, SRC_PARAM, SRC_RETURN \
        = get_datapoints()

    Xr, yr = load_data_tensors(RETURN_DATAPOINTS_X, RETURN_DATAPOINTS_Y, limit=-1)
    Xp, yp = load_data_tensors(PARAM_DATAPOINTS_X, PARAM_DATAPOINTS_Y, limit=-1)
    X = torch.cat((Xp, Xr))
    y = torch.cat((yp, yr))

    yp_src = np.load(SRC_PARAM, allow_pickle=True)
    yr_src = np.load(SRC_RETURN, allow_pickle=True)

    y_src = np.vstack((yp_src, yr_src))
    t2 = time.time_ns()
    print(t2 - t1)

    with open("./output/ml_inputs/label_encoder.pkl", 'rb') as f:
        encoder = pickle.load(f)
    _, config = load_m3()
    ort_session = ort.InferenceSession("./output/models/model_dltpy.onnx")
    t3 = time.time_ns()
    print(t3 - t2)

    data = torch.utils.data.TensorDataset(X, y)
    test_loader = torch.utils.data.DataLoader(data, batch_size=256)

    y_pred = evaluate_onnx(ort_session, test_loader, config, top_n=1)
    t4 = time.time_ns()
    print(t4 - t3)
    types_pred = np.array([encoder.classes_[type[0]] for type in y_pred])

    names = [src[2] for src in y_src]
    lines = [src[1] for src in y_src]

    df = pd.DataFrame(data={'name': names, 'line': lines, 'prediction': types_pred})
    df.to_csv('./output/predictions_tmp.csv')
    print(time.time_ns() - t4)

