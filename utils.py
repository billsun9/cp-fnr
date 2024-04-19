# '/manitou/pmg/projects/bys2107/cp-fnr'

import pandas as pd
import numpy as np
import random

from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import jaccard_score

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset
from models.baseline_mlp import *

def constructTrainCalTestSplit(X, Y, train_ratio=0.7, calibration_ratio=0.15, test_ratio=0.15, SEED=1000):
    """X,Y are numpy arrays, returns a train, validation, and test split"""
    assert train_ratio + calibration_ratio + test_ratio == 1
    
    indices = list(X.index)
    random.Random(SEED).shuffle(indices)
    
    train_split_point = int(len(indices) * train_ratio)
    calibration_split_point = train_split_point + int(len(indices) * calibration_ratio)
    train_indices = indices[:train_split_point]
    calibration_indices = indices[train_split_point:calibration_split_point]
    test_indices = indices[calibration_split_point:]
    
    X_train = X.loc[train_indices]
    Y_train = Y.loc[train_indices]
    X_calibration = X.loc[calibration_indices]
    Y_calibration = Y.loc[calibration_indices]
    X_test = X.loc[test_indices]
    Y_test = Y.loc[test_indices]
    
    X_train = X_train.values
    Y_train = Y_train.values
    X_calibration = X_calibration.values
    Y_calibration = Y_calibration.values
    X_test = X_test.values
    Y_test = Y_test.values
    
    # Print shapes for verification
    print("Loaded data splits")
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_calibration shape:", X_calibration.shape)
    print("Y_calibration shape:", Y_calibration.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)

    return X_train, Y_train, X_calibration, Y_calibration, X_test, Y_test

def eval_metrics(Y_test, prediction, verbose=True):
    metrics_dict = {}
    
    metrics_dict["Hamming Loss"] = hamming_loss(Y_test, prediction)
    metrics_dict["Accuracy Score"] = accuracy_score(Y_test, prediction)
    metrics_dict["F1 Score (micro)"] = f1_score(Y_test, prediction, average='micro')
    metrics_dict["F1 Score (macro)"] = f1_score(Y_test, prediction, average='macro')
    metrics_dict["Jaccard Score (average='samples')"] = jaccard_score(Y_test, prediction, average='samples')
    metrics_dict["Jaccard Score (average='macro')"] = jaccard_score(Y_test, prediction, average='macro')
    metrics_dict["Jaccard Score (average='micro')"] = jaccard_score(Y_test, prediction, average='micro')
    metrics_dict["Jaccard Score (average=None)"] = jaccard_score(Y_test, prediction, average=None)
    metrics_dict["Precision (macro)"] = precision_score(Y_test, prediction, average='macro')
    metrics_dict["Precision (micro)"] = precision_score(Y_test, prediction, average='micro')
    metrics_dict["Recall (micro)"] = recall_score(Y_test, prediction, average='micro')
    metrics_dict["Recall (macro)"] = recall_score(Y_test, prediction, average='macro')
    metrics_dict["Zero-One Loss (normalized)"] = zero_one_loss(Y_test, prediction, normalize=True)
    
    if verbose:
        for metric_name, value in metrics_dict.items():
            print(f"{metric_name}: {value}")

    return metrics_dict

def load_data_splits(SEED=1000):
    """Load the raw genomic data and antibiotic resistance data into numpy arrays"""
    X = pd.read_csv("data/cip_ctx_ctz_gen_multi_data.csv",index_col=0)
    Y = pd.read_csv("data/cip_ctx_ctz_gen_pheno.csv",index_col=0)
    X_train, Y_train, X_calibration, Y_calibration, X_test, Y_test = constructTrainCalTestSplit(X, Y, SEED=SEED)
    return X_train, Y_train, X_calibration, Y_calibration, X_test, Y_test

def fnr_weighted(preds, labels):
    """Calculates weighted FNR give two numpy arrays; Weighting is done based on frequency of positive labels"""
    # FNR for class i: FN / FN + TP
    classes = {}
    for j in range(len(preds[0])): # iterate over classes
        FN, TP = 0, 0
        for i in range(len(preds)): # iterate over exs
            if labels[i,j] == 1:
                if preds[i,j] != 1: FN += 1
                else: TP += 1
        if FN + TP > 0: # if label never appears, dont worry about it
            classes[j] = {"FN": FN, "TP": TP, "cnt": FN + TP}
    # print(classes)
    total = sum(classes[lbl]["cnt"] for lbl in classes)
    res = 0
    for lbl in classes: 
        res += (classes[lbl]["FN"] / total) # = (class_freq/total) * (FN/class_freq)
    return res

# Define your data loading process
def load_data(X, Y, device, batch_size=32):
    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    joined_dataset = TensorDataset(X, Y)
    return DataLoader(joined_dataset, batch_size=batch_size, shuffle=True)

def get_model(model_type, input_size, hidden_size, output_size):
    if model_type == 'mlp-mlc-sm':
        model = MlpMlcSm(input_size, hidden_size, output_size)
        model.name = 'mlp-mlc-sm'
    if model_type == 'mlp-mlc-md':
        model = MlpMlcMd(input_size, hidden_size, output_size)
        model.name = 'mlp-mlc-md'
    if model_type == 'mlp-mlc-lg':
        model = MlpMlcLg(input_size, hidden_size, output_size)
        model.name = 'mlp-mlc-lg'
    return model

def plotTrainValLosses(hist):
    epochs = range(1, len(hist[0]) + 1)

    plt.plot(epochs, hist[0], 'b', label='Training loss')
    plt.plot(epochs, hist[1], 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()