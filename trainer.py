import sys

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser

# noinspection PyUnresolvedReferences
from data_modules import *
from models import TSMixRegressorModule


class TSMixRegressorCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults({
            "early_stopping.monitor": "val_mse",
            "early_stopping.mode": "min",
            "early_stopping.patience": 20
        })


def cli_main(args=None):
    cb_learningrate = LearningRateMonitor(logging_interval='step')
    cb_modelchkpt = ModelCheckpoint(save_top_k=1, monitor="val_mse", mode="min")

    trainer_defaults = {
        'callbacks': [cb_learningrate, cb_modelchkpt],
        'max_epochs': 1000,
        'log_every_n_steps': 32,
        'enable_checkpointing': True,
    }

    run = (args is None and len(sys.argv) > 1 and sys.argv[1] in ['fit', 'validate', 'test', 'predict'])

    if run:
        cli = TSMixRegressorCLI(
            model_class=TSMixRegressorModule,
            run=True,
            trainer_defaults=trainer_defaults,
            args=args
        )
    else:
        cli = TSMixRegressorCLI(
            model_class=TSMixRegressorModule,
            run=False,
            trainer_defaults=trainer_defaults,
            args=args
        )
        print(f"Dataset: {cli.datamodule.__class__.__name__}")
        print(f"Fitting {cli.model.__class__.__name__} with: {dict(cli.config.model)}")
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)
        best_model = cli.model_class.load_from_checkpoint(cb_modelchkpt.best_model_path)
        print(f"Test:")
        cli.trainer.test(best_model, datamodule=cli.datamodule)


if __name__ == '__main__':
    cli_main()
