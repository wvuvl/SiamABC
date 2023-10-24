import os
import warnings
from typing import Dict, Any
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import hydra
import builtins
from hydra.utils import instantiate
from omegaconf import DictConfig

from core.train import get_tracking_datasets
# from core.train.AEVT_lightning_model import AEVTLightningModel
# from core.train.trainer import get_trainer
from core.train.train_val import AEVT_train_val
from core.utils import prepare_experiment, create_logger
from core.utils.torch_stuff import load_from_lighting, load_optimizer
logger = create_logger(__name__)
warnings.filterwarnings("ignore")


def train(gpu, ngpus_per_node, config: Dict[str, Any]) -> None:
    
    
    # suppress printing if not master
    if config["ddp"] and gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    if gpu is not None:
        print("Using GPU - {} for training".format(gpu))
    
    if config["ddp"]:
        if config["dist_url"] == "env://" and config["rank"] == -1:
            config["rank"] = int(os.environ["RANK"])
        if config["ddp"]:
            # mp distributed training, rank needs to be the  global rank among all the processes
            config["rank"] = config["rank"] * ngpus_per_node + gpu
        dist.init_process_group(backend=config["dist_backend"], init_method=config["dist_url"],
                                world_size=config["world_size"], rank=config["rank"])
        
        torch.distributed.barrier()
        
    model = instantiate(config["model"])
    print(model)
    train_dataset, val_dataset = get_tracking_datasets(config)
    # model = load_from_lighting(model, '/new_local_storage/zaveri/code/experiments/2023-10-16-01-38-23_Tracking_SiamABC_no_template_aug_resnet50_full/AEVT/trained_model_ckpt_11.pt')
    trainer = AEVT_train_val(model=model, config=config, train=train_dataset, val=val_dataset, ngpus_per_node=ngpus_per_node, gpu=gpu)
    # trainer.optimizer = load_optimizer(trainer.optimizer, '/new_local_storage/zaveri/code/experiments/2023-10-16-01-38-23_Tracking_SiamABC_no_template_aug_resnet50_full/AEVT/trained_model_ckpt_11.pt')
    train_loss, val_ios = trainer.train_network()


@hydra.main(config_name="AEVT_tracker", config_path="core/config")
def run_experiment(hydra_config: DictConfig) -> None:
    config = prepare_experiment(hydra_config)
    logger.info("Experiment dir %s" % config["experiment"]["folder"])
    
    save_path = os.path.join(config["experiment"]["folder"], config["experiment"]["name"])
    if os.path.exists(save_path) == False: os.makedirs(save_path)
    
    if config["dist_url"] == "env://" and config["world_size"] == -1:
        config["world_size"] = int(os.environ["WORLD_SIZE"])
        
    config["ddp"] = config["world_size"] > 1 or config["ddp"] == True
    
    # code inspired from simsiam
    ngpus_per_node = torch.cuda.device_count()
    if config["ddp"] and ngpus_per_node>1:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config["world_size"] = ngpus_per_node * config["world_size"]
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        train(config["gpus"][0], ngpus_per_node, config)
            

if __name__ == "__main__":
    run_experiment()