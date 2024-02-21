import os
import urllib.request

import numpy as np
from scipy.interpolate import interp1d
from .tsfile_data_module import NamedTSFileDataModule

URL_PPG_DALIA_TRAIN = 'https://zenodo.org/records/3902728/files/PPGDalia_TRAIN.ts'
URL_PPG_DALIA_TEST = 'https://zenodo.org/records/3902728/files/PPGDalia_TEST.ts'


class PPGDalia(NamedTSFileDataModule):
    seq_len = 512
    n_channels = 4

    def load_data(self, split):
        assert split in ['train', 'test']
        dir = os.path.join(self.data_dir, self.name)
        train_file = os.path.join(dir, f"{self.name}_TRAIN.ts")
        test_file = os.path.join(dir, f"{self.name}_TEST.ts")
        os.makedirs(dir, exist_ok=True)
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            if not self.download:
                raise FileNotFoundError(f"Dataset '{self.name}' not found - use download option to download it")
            urllib.request.urlretrieve(URL_PPG_DALIA_TRAIN, train_file)
            urllib.request.urlretrieve(URL_PPG_DALIA_TEST, test_file)
        tsfile = train_file if split == 'train' else test_file
        X, y = [], []
        in_header = True
        with (open(tsfile) as f):
            for line in f:
                line = line.strip().lower()
                if in_header:
                    in_header = (line != '@data')
                    continue
                line = line.replace("?", "NaN")
                data = line.split(":")

                x = np.zeros(shape=(self.n_channels, self.seq_len))
                for i, seriesdata in enumerate(data[:-1]):
                    seq = np.array([float(v) for v in seriesdata.split(",")])
                    if np.isnan(seq).any():
                        idx = np.arange(len(seq))
                        notnans = np.where(~np.isnan(seq))[0]
                        nans = np.where(np.isnan(seq))[0]
                        f = interp1d(idx[notnans], seq[notnans], kind='linear',
                                     fill_value='extrapolate', bounds_error=False)
                        seq[nans] = f(nans)
                    lx = len(seq)
                    if lx < self.seq_len:
                        seq = [seq[int(i * lx / self.seq_len)] for i in range(self.seq_len)]
                    if lx > self.seq_len:
                        seq = seq[-self.seq_len:]
                    x[i, :] = seq
                X.append(x)
                y.append(float(data[-1]))

        return np.array(X), np.array(y)
