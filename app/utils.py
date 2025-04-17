import pandas as pd
import numpy as np
import random
import os
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def seed_everything(seed=42):
    """
    Ensure reproducibility.
    :param seed: Integer defining the seed number.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_and_process_data(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df.iloc[:, 4:-1]  # drop location metadata (first 4 columns) and last column with NaNs

    daily_cases = df.sum(axis=0)
    daily_cases.index = pd.to_datetime(daily_cases.index, format="%m/%d/%y")
    daily_cases = daily_cases.diff().fillna(daily_cases.iloc[0]).astype(np.int64)

    return daily_cases

def normalize_data(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(np.expand_dims(data, axis=1))
    return data_scaled, scaler

def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data)-seq_length-1):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

def prepare_data(dataset_path, seq_length=5, test_size=0.2, seed=42):
    daily_cases = load_and_process_data(dataset_path)
    normalized_data, _ = normalize_data(daily_cases)

    # Split daily cases by time
    split_idx = int(len(normalized_data) * (1 - test_size))
    train_series = normalized_data[:split_idx]
    test_series = normalized_data[split_idx - seq_length:]  # Include some history by an overlap of seq_length between train and test set
    #test_series = normalized_data[split_idx:] # No overlap

    # Create sequences (sliding windows)
    X_train, y_train = create_sequences(train_series, seq_length)
    X_test, y_test = create_sequences(test_series, seq_length)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    return X_train, y_train, X_test, y_test
