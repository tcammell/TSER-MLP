import os

import numpy as np

from .Monash_UEA_UCR_Regression_Archive_datamodules import HouseholdPowerConsumption1, HouseholdPowerConsumption2
from .tsfile_data_module import NamedTSFileDataModule
from .utils import read_tsfile_with_time_features, interpolate_nans_tsdataset


class HouseholdPowerConsumption1_TS(NamedTSFileDataModule):
    seq_len = 1440
    n_channels = 9

    def load_data(self, split: str):
        ds = HouseholdPowerConsumption1()
        ds.load_data(split)  # download etc.
        dir = os.path.join(self.data_dir, ds.name)
        file = os.path.join(dir, f"{ds.name}_{split.upper()}.ts")
        X, y = read_tsfile_with_time_features(file)
        if self.interpolate:
            X, y = interpolate_nans_tsdataset(X, y)
        return np.array(X), np.array(y)


class HouseholdPowerConsumption2_TS(NamedTSFileDataModule):
    seq_len = 1440
    n_channels = 9

    def load_data(self, split: str):
        ds = HouseholdPowerConsumption2()
        ds.load_data(split)  # download etc.
        dir = os.path.join(self.data_dir, ds.name)
        file = os.path.join(dir, f"{ds.name}_{split.upper()}.ts")
        X, y = read_tsfile_with_time_features(file)
        if self.interpolate:
            X, y = interpolate_nans_tsdataset(X, y)
        return np.array(X), np.array(y)
