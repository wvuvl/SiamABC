import warnings
from typing import Dict, Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from train import get_tracking_datasets
from train.AEVT_lightning_model import AEVTLightningModel
from train.trainer import get_trainer
from utils import prepare_experiment, create_logger

logger = create_logger(__name__)
warnings.filterwarnings("ignore")


def train(config: Dict[str, Any]) -> None:
    model = instantiate(config["model"])
    train_dataset, val_dataset = get_tracking_datasets(config)
    model = AEVTLightningModel(model=model, config=config, train=train_dataset, val=val_dataset)
    trainer = get_trainer(config=config)
    trainer.fit(model)


@hydra.main(config_name="fear_tracker", config_path="config")
def run_experiment(hydra_config: DictConfig) -> None:
    config = prepare_experiment(hydra_config)
    logger.info("Experiment dir %s" % config["experiment"]["folder"])
    train(config)


if __name__ == "__main__":
    run_experiment()