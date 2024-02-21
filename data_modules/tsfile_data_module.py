from typing import Optional

import lightning as L
import torch
from aeon.datasets import load_regression
from aeon.datasets.dataset_collections import get_downloaded_tsc_tsr_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from .utils import interpolate_nans_tsdataset, fit_ts_scaler


class TSFileDataModule(L.LightningDataModule):

    n_channels: int
    seq_len: int

    def __init__(
            self,
            name: str,
            val_split: Optional[float] = 0.0,
            batch_size: Optional[int] = 32,
            num_workers: Optional[int] = 8,
            interpolate: Optional[bool] = True,
            shuffle: Optional[bool] = True,
            drop_last: Optional[bool] = False,
            scale: bool = True,
            data_dir: str = 'data',
            download: bool = True,
    ):
        super().__init__()
        self.name = name
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.interpolate = interpolate
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.scale = scale
        self.data_dir = data_dir
        self.download = download

    def load_data(self, split):
        if not self.download and self.name not in get_downloaded_tsc_tsr_datasets(extract_path=self.data_dir):
            raise FileNotFoundError(f"Dataset '{self.name}' not found - use download option to download it")
        X, y = load_regression(self.name, split=split, extract_path=self.data_dir)
        if self.interpolate:
            X, y = interpolate_nans_tsdataset(X, y)

        return X, y

    def setup(self, stage: str):

        train_x, train_y = self.load_data('train')
        test_x, test_y = self.load_data('test')

        if 0 < self.val_split < 1:
            train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=self.val_split)
        elif self.val_split == 0:
            val_x, val_y = test_x, test_y
        else:
            val_x, val_y = train_x, train_y

        if self.scale:
            transform = fit_ts_scaler(StandardScaler(), train_x)
            train_x = transform(train_x)
            val_x = transform(val_x)
            test_x = transform(test_x)

        train_x = torch.tensor(train_x).float()
        val_x = torch.tensor(val_x).float()
        test_x = torch.tensor(test_x).float()

        train_y = torch.tensor(train_y).reshape(-1, 1).float()
        val_y = torch.tensor(val_y).reshape(-1, 1).float()
        test_y = torch.tensor(test_y).reshape(-1, 1).float()

        self.train_ds = TensorDataset(train_x, train_y)
        self.val_ds = TensorDataset(val_x, val_y)
        self.test_ds = TensorDataset(test_x, test_y)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            persistent_workers=True
        )

    def predict_dataloader(self):
        raise NotImplementedError

    def teardown(self, stage: str):
        pass


class NamedTSFileDataModule(TSFileDataModule):
    def __init__(self, *args, **kwargs):
        name = self.__class__.__name__
        super().__init__(name=name, *args, **kwargs)
