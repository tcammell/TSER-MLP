import argparse
import functools
import os
import sys

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

import optuna
from data_modules import *
from models import TSMixRegressorModule

DEFAULT_PARAMS = dict(
    # model params (ranges)
    n_block=[1, 2, 4, 6],
    hidden_size=[128, 256, 512],
    dropout=[0.2, 0.33, 0.5],
    revin=[True, False],
    gmlp_proj=[True, False],
    gmlp_blocks=[1, 2, 4, 6],
    gmlp_patch_size=[1],
    gmlp_d_model=[256],
    gmlp_d_ffn=[256, 512],
    lr=[5e-4],
    lr_patience=[10],

    # trainer params
    early_stopping_patience=40,
    max_epochs=1000,

    # data params
    val_split=0.33,
    num_workers=8,
    batch_size=32,
    scale=True,
)

PARAMS = {
    # per-dataset overrides
    AppliancesEnergy: dict(
        lr_patience=[20],
        early_stopping_patience=80,
    ),
    AustraliaRainfall: dict(
    ),
    BIDMC32HR: dict(
        lr_patience=[3],
        early_stopping_patience=12,
    ),
    BIDMC32RR: dict(
        lr_patience=[3],
        early_stopping_patience=12,
    ),
    BIDMC32SpO2: dict(
    ),
    BeijingPM10Quality: dict(
    ),
    BeijingPM25Quality: dict(
    ),
    BenzeneConcentration: dict(
    ),
    Covid3Month: dict(
        lr_patience=[100],
        early_stopping_patience=400,
        dropout=[0.2, 0.5, 0.9, 0.99, 0.995],
    ),
    FloodModeling1: dict(
        lr_patience=[100],
        early_stopping_patience=400,
    ),
    FloodModeling2: dict(
        lr_patience=[100],
        early_stopping_patience=400,
    ),
    FloodModeling3: dict(
        lr_patience=[100],
        early_stopping_patience=400,
    ),
    HouseholdPowerConsumption1: dict(
        lr_patience=[10],
        early_stopping_patience=12,
    ),
    HouseholdPowerConsumption2: dict(
        lr_patience=[10],
        early_stopping_patience=12,
    ),
    IEEEPPG: dict(
        lr_patience=[3],
        early_stopping_patience=12,
    ),
    LiveFuelMoistureContent: dict(
    ),
    NewsHeadlineSentiment: dict(
        lr_patience=[3],
        early_stopping_patience=12,
    ),
    NewsTitleSentiment: dict(
        lr_patience=[3],
        early_stopping_patience=12,
    ),
    PPGDalia: dict(
    ),
}


def objective(
        trial: optuna.trial.Trial,
        datamodule: TSFileDataModule,
        params: dict,
        seed: int,
) -> float:
    if seed:
        L.seed_everything(seed)
    checkpoint_filename = f'trial_{datamodule.name}_{trial.number}'
    cb_checkpoint = ModelCheckpoint(filename=checkpoint_filename)

    cb_earlystop = EarlyStopping(monitor="val_rmse", mode="min", patience=params['early_stopping_patience'])
    cb_learningrate = LearningRateMonitor(logging_interval='step')
    cb_pruning = PyTorchLightningPruningCallback(trial, monitor='val_rmse')

    trainer = L.Trainer(
        callbacks=[cb_earlystop, cb_learningrate, cb_checkpoint, cb_pruning],
        max_epochs=params['max_epochs'],
        log_every_n_steps=8,
        enable_checkpointing=True,
        logger=True,
    )

    n_block = trial.suggest_categorical('n_block', params['n_block'])
    ch_in = trial.suggest_categorical('ch_in', [datamodule.n_channels])
    seq_len = trial.suggest_categorical('seq_len', [datamodule.seq_len])
    hidden_size = trial.suggest_categorical('hidden_size', params['hidden_size'])
    activation = trial.suggest_categorical('activation', ['relu'])
    dropout = trial.suggest_categorical('dropout', params['dropout'])
    revin = trial.suggest_categorical('revin', params['revin'])

    if datamodule.n_channels == 1:
        univariate = trial.suggest_categorical('univariate', [True, False])
    else:
        univariate = trial.suggest_categorical('univariate', [False])

    gmlp_proj = trial.suggest_categorical('gmlp_proj', params['gmlp_proj'])
    if gmlp_proj:
        gmlp_blocks = trial.suggest_categorical('gmlp_blocks', params['gmlp_blocks'])
        gmlp_patch_size = trial.suggest_categorical('gmlp_patch_size', params['gmlp_patch_size'])
        gmlp_d_model = trial.suggest_categorical('gmlp_d_model', params['gmlp_d_model'])
        gmlp_d_ffn = trial.suggest_categorical('gmlp_d_ffn', params['gmlp_d_ffn'])
    else:
        gmlp_blocks = 0
        gmlp_patch_size = 0
        gmlp_d_model = 0
        gmlp_d_ffn = 0

    lr = trial.suggest_categorical('lr', params['lr'])
    lr_patience = trial.suggest_categorical('lr_patience', params['lr_patience'])

    model = TSMixRegressorModule(
        n_block=n_block,
        ch_in=ch_in,
        seq_len=seq_len,
        hidden_size=hidden_size,
        activation=activation,
        dropout=dropout,
        univariate=univariate,
        revin=revin,
        gmlp_proj=gmlp_proj,
        gmlp_blocks=gmlp_blocks,
        gmlp_patch_size=gmlp_patch_size,
        gmlp_d_model=gmlp_d_model,
        gmlp_d_ffn=gmlp_d_ffn,
        lr=lr,
        lr_patience=lr_patience,
    )

    try:
        trainer.fit(model, datamodule)
        curr_loss = trainer.callback_metrics['val_rmse'].item()
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"EXCEPTION: {e}")
        curr_loss = float('inf')

    return curr_loss


def tune(dataset, n_trials, optuna_dir, seed):
    dm_class = getattr(sys.modules[__name__], dataset)
    params = dict()
    params.update(DEFAULT_PARAMS)
    params.update(PARAMS[dm_class])

    datamodule = dm_class(
        num_workers=params['num_workers'],
        batch_size=params['batch_size'],
        scale=params['scale'],
        val_split=params['val_split'],
    )

    os.makedirs(optuna_dir, exist_ok=True)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(
        study_name=f"{datamodule.name}",
        direction="minimize",
        pruner=pruner,
        storage=f"sqlite:///{optuna_dir}/{datamodule.name}_optuna.db", load_if_exists=True
    )

    obj = functools.partial(objective, datamodule=datamodule, params=params, seed=seed)
    study.optimize(obj, n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    print("  Full Summary of Trials:  ")
    print(study.trials_dataframe())

    # plot_optimization_history(study).show()
    # plot_intermediate_values(study).show()
    # try:
    #     plot_param_importances(study).show()
    # except ValueError:
    #     pass


def cli():
    datasets = [d.__name__.split('.')[-1] for d in DATASETS]

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=datasets, required=True, help='Dataset')
    parser.add_argument('-n', '--trials', type=int, default=0, help='Number of trials to run')
    parser.add_argument('-o', '--optuna_dir', type=str, default='optuna',
                        help='Directory to store optuna sqlite database, default ./optuna')
    parser.add_argument('-s', '--seed', type=int, default=None, help='random seed, default None')
    args = parser.parse_args()
    tune(args.dataset, args.trials, args.optuna_dir, args.seed)


if __name__ == '__main__':
    cli()
