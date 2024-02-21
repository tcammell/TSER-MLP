from .tsfile_data_module import TSFileDataModule
from .Monash_UEA_UCR_Regression_Archive_datamodules import *
from .HouseholdPowerConsumption_TS import HouseholdPowerConsumption1_TS, HouseholdPowerConsumption2_TS

DATASETS = [
    AppliancesEnergy,
    AustraliaRainfall,
    BIDMC32HR,
    BIDMC32RR,
    BIDMC32SpO2,
    BeijingPM10Quality,
    BeijingPM25Quality,
    BenzeneConcentration,
    Covid3Month,
    FloodModeling1,
    FloodModeling2,
    FloodModeling3,
    HouseholdPowerConsumption1,
    HouseholdPowerConsumption2,
    IEEEPPG,
    LiveFuelMoistureContent,
    NewsHeadlineSentiment,
    NewsTitleSentiment,
    PPGDalia,
]


