import numpy as np
import pandas as pd                 # Import au niveau module (obligatoire)
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os

TICKER          = "TSLA"
TARGET_COL      = "High"
SEQUENCE_LENGTH = 60
BATCH_SIZE      = 32
TEST_SPLIT      = 0.2
VAL_SPLIT       = 0.1

def download_stock_data():
    csv_path = "data/TSLA_2019_2024.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.dropna(inplace=True)
    print(f"[data_loader_rnn] {len(df)} jours charges")
    return df

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length])
    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y)
    return X, y

def load_stock_data():
    df     = download_stock_data()
    prices = df[TARGET_COL].values.astype('float32')

    print(f"[data_loader_rnn] Prix min=${prices.min():.2f} max=${prices.max():.2f}")

    scaler        = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    X, y = create_sequences(prices_scaled, SEQUENCE_LENGTH)

    n         = len(X)
    train_end = int(n * (1 - TEST_SPLIT - VAL_SPLIT))
    val_end   = int(n * (1 - TEST_SPLIT))

    X_train, y_train = X[:train_end],         y[:train_end]
    X_val,   y_val   = X[train_end:val_end],  y[train_end:val_end]
    X_test,  y_test  = X[val_end:],           y[val_end:]

    print(f"[data_loader_rnn] Train={X_train.shape[0]} Val={X_val.shape[0]} Test={X_test.shape[0]}")

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds   = tf.data.Dataset.from_tensor_slices((X_val,   y_val  )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds  = tf.data.Dataset.from_tensor_slices((X_test,  y_test )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, scaler, df, X_test, y_test
