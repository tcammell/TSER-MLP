# noinspection PyUnresolvedReferences
from .ppg_dalia import PPGDalia
from .tsfile_data_module import NamedTSFileDataModule


class AppliancesEnergy(NamedTSFileDataModule):
    seq_len = 144
    n_channels = 24


class AustraliaRainfall(NamedTSFileDataModule):
    seq_len = 24
    n_channels = 3


class BIDMC32HR(NamedTSFileDataModule):
    seq_len = 4000
    n_channels = 2


class BIDMC32RR(NamedTSFileDataModule):
    seq_len = 4000
    n_channels = 2


class BIDMC32SpO2(NamedTSFileDataModule):
    seq_len = 4000
    n_channels = 2


class BeijingPM10Quality(NamedTSFileDataModule):
    seq_len = 24
    n_channels = 9


class BeijingPM25Quality(NamedTSFileDataModule):
    seq_len = 24
    n_channels = 9


class BenzeneConcentration(NamedTSFileDataModule):
    seq_len = 240
    n_channels = 8


class Covid3Month(NamedTSFileDataModule):
    seq_len = 84
    n_channels = 1


class FloodModeling1(NamedTSFileDataModule):
    seq_len = 266
    n_channels = 1


class FloodModeling2(NamedTSFileDataModule):
    seq_len = 266
    n_channels = 1


class FloodModeling3(NamedTSFileDataModule):
    seq_len = 266
    n_channels = 1


class HouseholdPowerConsumption1(NamedTSFileDataModule):
    seq_len = 1440
    n_channels = 5


class HouseholdPowerConsumption2(NamedTSFileDataModule):
    seq_len = 1440
    n_channels = 5


class IEEEPPG(NamedTSFileDataModule):
    seq_len = 1000
    n_channels = 5


class LiveFuelMoistureContent(NamedTSFileDataModule):
    seq_len = 365
    n_channels = 7


class NewsHeadlineSentiment(NamedTSFileDataModule):
    seq_len = 144
    n_channels = 3


class NewsTitleSentiment(NamedTSFileDataModule):
    seq_len = 144
    n_channels = 3
