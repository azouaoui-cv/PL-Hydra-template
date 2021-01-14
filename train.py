# Standard library
import logging
import os

# Config related modules
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
# PTL
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

# Custom modules
from mypackage import BoringModel, DataModule

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train(cfg):
    # Seed everything
    seed_everything(cfg.seed)
    # Load datamodule
    data = DataModule(cfg)
    # Callbacks
    callbacks = [ModelCheckpoint(**cfg.checkpoint)]
    # Load model
    model = BoringModel(cfg)
    # Instanciate trainer
    trainer = pl.Trainer(callbacks=callbacks, **cfg.trainer.params)
    # Fit trainer
    trainer.fit(model, datamodule=data)
    logger.info(trainer.test())


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(f"Current working directory: {os.getcwd()}")
    try:
        train(cfg)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()
