"""

Modified from
Main Author of this file: FEAR
Repo: https://github.com/PinataFarms/FEARTracker/tree/main
File: https://github.com/PinataFarms/FEARTracker/blob/main/model_training/train

"""

from typing import Dict

from torch.utils.data import ConcatDataset

from core.utils import create_logger
from core.train.dataloader import TrackingDataset, SequenceDatasetWrapper

logger = create_logger(__name__)

def get_tracking_test_datasets(config):
    test_datasets = []
    if "datasets" in config["test"]:
        for dataset_config in config["test"]["datasets"]:
            ds = SequenceDatasetWrapper.from_config(dataset_config)
            logger.info("Valid dataset %s %d", str(ds), len(ds))
            test_datasets.append(ds)
        return ConcatDataset(test_datasets)
   
   

def get_tracking_datasets(config) -> [ConcatDataset, ConcatDataset]:
    train_datasets = []
    for dataset_config in config["train"]["datasets"]:
        cls = TrackingDataset
        ds = cls.from_config(dict(dataset=dataset_config, tracker=config["tracker"]))
        logger.info("Train dataset %s %d", str(ds), len(ds))
        train_datasets.append(ds)

    val_datasets = []
    if "datasets" in config["val"]:
        for dataset_config in config["val"]["datasets"]:
            ds = SequenceDatasetWrapper.from_config(dataset_config)
            logger.info("Valid dataset %s %d", str(ds), len(ds))
            val_datasets.append(ds)
    
    return ConcatDataset(train_datasets), (ConcatDataset(val_datasets) if len(val_datasets)>0 else None)


__all__ = [
    "TrackingDataset",
    "get_tracking_dataset",
    "get_tracking_datasets",
]
