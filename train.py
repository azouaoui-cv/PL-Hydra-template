# Standard library
import pdb
import os
import logging
# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# PTL
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
# Config related modules
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
# Custom modules
from mypackage import DataModule, BoringModel

def train(cfg):
    # Seed everything
    seed_everything(cfg.seed)
    # Load datamodule
    data = DataModule(cfg)
    # Callbacks
    callbacks = [ModelCheckpoint(**cfg.checkpoint)]
    # Load model
    model = BoringModel(cfg)
    # Check for resume
    if cfg.resume:
        logging.info(f"Attempting to resume training")
        # Get Hydra config
        hydra_cfg = HydraConfig.get()
        run_dir = os.path.join(hydra_cfg.runtime.cwd, hydra_cfg.run.dir)
        assert run_dir == os.getcwd()
        # Get checkpoint path
        ckpt_dir = cfg.checkpoint.dirpath
        ckpt_path = os.path.join(run_dir, ckpt_dir, "last.ckpt")
        if os.path.exists(ckpt_path):
            logging.info(f"Loading existing ckpt @ {ckpt_path}")
            # Reload trainer
            trainer = pl.Trainer(resume_from_checkpoint=ckpt_path, 
				callbacks=callbacks, 
				**cfg.trainer.params)
        else:
            logging.info(f"No existing ckpt found. Training from scratch")
            # Setup trainer
            trainer = pl.Trainer(callbacks=callbacks, **cfg.trainer.params)
    else:
        logging.info(f"Training from scratch")
        # Setup trainer
        trainer = pl.Trainer(callbacks=callbacks, **cfg.trainer.params)
    # Fit trainer
    trainer.fit(model, datamodule=data)
    print(trainer.test())

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(f"Current working directory: {os.getcwd()}")
    try:
        train(cfg)
    except Exception as e:
        logger.critical(e, exc_info=True)

if __name__ == "__main__":
    main()
