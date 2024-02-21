import os
import re
import urllib

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from data_modules.timefeatures import time_features


def interpolate_nans_tsdataset(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = X.copy()
    n_data, n_features, n_samples = X.shape
    keep_idx = set(range(n_data))
    series_t = np.arange(n_samples)
    for i in range(n_data):
        for j in range(n_features):
            series = X[i, j, :]
            if np.isnan(series).any():
                notnan = ~np.isnan(series)
                nc = sum(notnan)
                if nc == 0:
                    keep_idx.remove(i)
                    break
                x = np.where(notnan)[0]
                if nc == 1:
                    X[i, j, :] = series[x]
                else:
                    f = interp1d(x, series[x], kind='linear', fill_value='extrapolate')
                    interpolated_values = f(series_t)
                    X[i, j, :] = interpolated_values
    X, y = X[list(keep_idx)], y[list(keep_idx)]
    return X, y


def max_seq_len(X):
    batch_size, n_ch, _ = X.shape
    max_len = 0
    for i in range(batch_size):
        for j in range(n_ch):
            max_len = max(max_len, len(X[i, j]))
    return max_len


def scale_seq_length(X):
    X = X.copy()
    batch_size, n_ch, _ = X.shape
    max_len = 0
    updates = 0
    for i in range(batch_size):
        for j in range(n_ch):
            _len = len(X[i, j])
            if _len != max_len:
                max_len = max(max_len, _len)
                updates += 1

    if updates > 1:
        print("SCALING SEQUENCE LENGTHS")
        for i in range(batch_size):
            for j in range(n_ch):
                seq_len = len(X[i, j])
                if seq_len != max_len:
                    X[i, j] = [X[i, j, int(k * seq_len / max_len)] for k in range(max_len)]
    return X


def fit_ts_scaler(scaler, X: np.ndarray):
    n_batch, n_channel, seq_len = X.shape
    scaler.fit(X.reshape(-1, n_channel))

    def transform(_X):
        nb, nc, ns = _X.shape
        return scaler.transform(_X.reshape(-1, nc)).reshape(nb, nc, ns)

    return transform


def read_tsfile_header(tsfile):
    pat = re.compile(r'@(\w+) (.*)(#|$)')
    map_bool = lambda x: {"true": True, "false": False}[x]
    map_int = lambda x: int(x)
    map_classlabel = lambda x: x.split(" ")[1:] if x.startswith("true") else []
    tags = {
        "timestamps": map_bool,
        "missing": map_bool,
        "univariate": map_bool,
        "dimension": map_int,
        "equallength": map_bool,
        "serieslength": map_int,
        "targetlabel": map_bool,
        "classlabel": map_classlabel,
    }
    meta = {}
    with (open(tsfile, encoding="utf-8") as f):
        while True:
            line = f.readline().partition('#')[0].strip()
            if line == '@data':
                break
            m = pat.match(line)
            if m:
                tag, value = m[1], m[2]
                if tag in tags:
                    value = tags[tag](value)
                meta.update({tag: value})
    return meta


def download_tsdataset(name: str, url: str, split: str, data_dir: str = 'data'):
    assert split in ['train', 'test']
    dir = os.path.join(data_dir, name)
    filename = os.path.join(dir, f"{name}_{split.upper()}.ts")
    os.makedirs(dir, exist_ok=True)
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    return filename


def read_tsfile_with_time_features(filename, format: str = '%Y-%m-%d %H:%M:%S', freq='h'):
    pat = re.compile(r'\(([^(]+)\)([,:])')
    with open(filename) as f:
        while True:
            line = f.readline().strip()
            if line == '@data':
                break
        seq = 0
        X = []
        y = []
        while True:
            seq_df = pd.DataFrame()
            feat = 0
            ptr = 0
            line = f.readline().strip()
            if not line:
                break
            while line[ptr] == '(':
                records = []
                while True:
                    m = pat.match(line[ptr:])
                    ptr += len(m[0])
                    dt, val = m[1].split(',')
                    try:
                        val = float(val)
                    except ValueError:
                        val = np.nan
                    records.append({'dt': dt, f'f{feat}': val})
                    sep = m[2]
                    if sep == ':':
                        break
                df_feat = pd.DataFrame.from_records(records)
                df_feat['dt'] = pd.to_datetime(df_feat['dt'], format=format)
                df_feat.set_index('dt', inplace=True)

                if len(seq_df):
                    seq_df = seq_df.join(df_feat)
                else:
                    seq_df = df_feat
                feat += 1

            y.append(float(line[ptr:]))

            # add time features
            tf = time_features(seq_df.index, freq=freq)
            for i in range(tf.shape[0]):
                seq_df[f"t{i}"] = tf[:][i]

            X.append(seq_df.reset_index(drop=True).to_numpy())

    return np.stack(X).transpose(0,2,1), np.stack(y)


import unittest

class Test(unittest.TestCase):
    def test_time_features(self):
        ds = 'HouseholdPowerConsumption1'
        test_x, test_y = read_tsfile_with_time_features(f'../data/{ds}/{ds}_TEST.ts')
        print(test_x.shape, test_y.shape)
