import os
from typing import Dict, Tuple, Any, List, Union, Callable, Iterable
from tqdm import trange, tqdm
import time
import numpy as np
import torch
from hydra.utils import instantiate
from torch import Tensor, inference_mode, load, save
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler
from torch.utils.data.dataloader import default_collate
from torchmetrics import MetricCollection
from torchvision.ops import box_convert
from statistics import mean
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn

from core.metrics import DatasetAwareMetric, BoxIoUMetric, TrackingFailureRateMetric, box_iou_metric
from core.models.loss import AEVTLoss
from core.utils.box_coder import TrackerDecodeResult, AEVTBoxCoder
from core.utils.utils import read_img, get_iou, plot_loss

from core.utils.logger import create_logger
import core.constants as constants


logger = create_logger(__name__)

def get_collate_for_dataset(dataset: Union[Dataset, ConcatDataset]):
    """
    Returns collate_fn function for dataset. By default, default_collate returned.
    If the dataset has method get_collate_fn() we will use it's return value instead.
    If the dataset is ConcatDataset, we will check whether all get_collate_fn() returns
    the same function.

    Args:
        dataset: Input dataset

    Returns:
        Collate function to put into DataLoader
    """
    collate_fn = default_collate

    if hasattr(dataset, "get_collate_fn"):
        collate_fn = dataset.get_collate_fn()

    if isinstance(dataset, ConcatDataset):
        collates = [get_collate_for_dataset(ds) for ds in dataset.datasets]
        if len(set(collates)) != 1:
            raise ValueError("Datasets have different collate functions")
        collate_fn = collates[0]
    return collate_fn

class AEVT_train_val:
    input_type = torch.float32
    target_type = torch.float32
    def __init__(self, model: torch.nn.Module, 
                config: Dict[str, Any], 
                train: Dataset,
                val: Dataset,
                ngpus_per_node:int,
                gpu:int,
                pretrained=None
                ):
        super().__init__()
        
        
        
        self.config = config
        self.use_ddp = self.config["ddp"]
        # create model and move it to GPU with id rank
        self.device_id = gpu
        if self.config["sync_bn"]: model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        if self.device_id is not None:
            torch.cuda.set_device(self.device_id)
            model = model.to(self.device_id)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            self.config["batch_size"]["train"] = int(self.config["batch_size"]["train"] / ngpus_per_node)
            self.config["num_workers"] = int((self.config["num_workers"] + ngpus_per_node - 1) / ngpus_per_node)
            self.model = DDP(model, device_ids=[self.device_id]) #,find_unused_parameters=True)
        else:
            model = model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            self.model = DDP(model)
                
        
    
    
        
        self.train_dataset = train
        self.val_dataset = val
        self.epoch_num = 0
        self.tensorboard_logger = None
        
        
        self.tracker = instantiate(config["tracker"], model=self.model.module, cuda_id=self.device_id)
        self.box_coder = self.tracker.box_coder
        self.metrics = MetricCollection(
            {
                "box_iou": BoxIoUMetric(compute_on_step=True),
                "failure_rate": TrackingFailureRateMetric(compute_on_step=True),
            }
        )
        self.dataset_aware_metric = DatasetAwareMetric(metric_name="box_iou", metric_fn=box_iou_metric)
        self.criterion = AEVTLoss(coeffs=config["loss"]["coeffs"]).to(self.device_id)
        
        
        self.train_dl, self.train_sampler = self._get_dataloader(self.train_dataset, "train")
        self.val_dl, self.val_sampler = self._get_dataloader(self.val_dataset, "val") if self.val_dataset is not None else (None,None)
        
        
        self.configure_optimizers()
        self.start_epoch = 0
        self.max_epochs = self.config["max_epochs"]
        
        self.save_path = os.path.join(config["experiment"]["folder"], config["experiment"]["name"])
        
        
        cudnn.benchmark = True
        
    def _get_batch_size(self, mode: str = "train"):
        if isinstance(self.config["batch_size"], dict):
            return self.config["batch_size"][mode]
        return self.config["batch_size"]
    
    def _get_dataloader(self, dataset: Dataset, loader_name: str):
        """
        Instantiate DataLoader for given dataset w.r.t to config and mode.
        It supports creating a custom sampler.
        Note: For DDP mode, we support custom samplers, but trainer must be called with:
            >>> replace_sampler_ddp=False

        Args:
            dataset: Dataset instance
            config: Dataset config
            loader_name: Loader name (train or val)

        Returns:

        """
        collate_fn = get_collate_for_dataset(dataset)

        drop_last = loader_name == "train"

        sampler = DistributedSampler(dataset) if self.use_ddp else None
    
        should_shuffle = (sampler is None) and (loader_name == "train")
        batch_size = self._get_batch_size(loader_name)
        # Number of workers must not exceed batch size
        num_workers = min(batch_size, self.config["num_workers"])
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=sampler,
            num_workers=num_workers if loader_name == "train" else 0,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )
        return loader, sampler
    
    def configure_optimizers(self):
        print('Learning Rate - Set: ', 1e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.get('lr', 0.0001), momentum=0.9, weight_decay=1e-05)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                        mode=self.config.get("metric_mode", "min"),
                                                        patience=2,
                                                        factor=0.5,
                                                        verbose=True,
                                                        min_lr=1e-6)
    
    
    def train_network(self):  
        
        # for g in self.optimizer.param_groups:
        #     g['lr'] = 1e-6
    
        train_losses = []
        val_ious = {}
        start_train = time.time()
        logger.info('Beginning network training.\n')
        for e in range(self.start_epoch+1, self.max_epochs + 1):
            
            if self.use_ddp: self.train_sampler.set_epoch(e)
            
            logger.info("Resampling Dataset(s) -- ")
            datasets_to_update = self.train_dl.dataset.datasets if type(self.train_dl.dataset) is ConcatDataset else [self.train_dl.dataset]
            for dataset in datasets_to_update:
                dataset.resample()
                if e > 15: # and len(datasets_to_update) > 1 : 
                    # if dataset.item_sampler.dynamic_frame_offset<30: dataset.item_sampler.dynamic_frame_offset+=5
                    if dataset.item_sampler.frame_offset<150: dataset.item_sampler.frame_offset+=5
                # logger.info(f'Dynamic frame offset for dynamic search region={dataset.item_sampler.dynamic_frame_offset}')
                logger.info(f'Dynamic frame offset for dynamic template={dataset.item_sampler.frame_offset}')
                
            
                    
            logger.info(f"lr={self.optimizer.param_groups[0]['lr']}")
            
            train_epoch_loss, class_loss, regression_loss, search_sim_loss, dynamic_sim_loss, dissim_loss = self.train_epoch(e, self.train_dl)
            self.scheduler.step(train_epoch_loss)
            train_losses.append(train_epoch_loss)
            self.save_network_checkpoint(e)


            logger.info('Train loss: {:.3f} \n'.format(train_epoch_loss))

            logger.info('Specific Train losses - \nClass Loss: {:.3f}; \nRegression Loss: {:.3f} \nSearch Sim Loss: {:.3f} \nDynamic Sim Loss: {:.3f} \nDiss Sim Loss: {:.3f} \n'\
                    .format(class_loss, regression_loss, search_sim_loss, dynamic_sim_loss, dissim_loss))
            

            if e%5==0:
                if self.val_dl is not None:
                    if self.use_ddp: self.val_sampler.set_epoch(e)
                    val_iou = self.validate_network(self.val_dl)
                    # logger.info('Train loss: {:.3f} \n'.format(train_epoch_loss))

                    # logger.info('Specific Train losses - \nClass Loss: {:.3f}; \nRegression Loss: {:.3f} \nSearch Sim Loss: {:.3f} \nDynamic Sim Loss: {:.3f} \n'\
                    #         .format(class_loss, regression_loss, search_sim_loss, dynamic_sim_loss))
                    logger.info('Validation Iou: {} \n'.format(val_iou))
                    for val_key in val_iou.keys():
                        if val_key not in val_ious.keys():
                            val_ious[val_key] = []
                        val_ious[val_key].append(val_iou[val_key]) 

            # if (e%10)==0 or e==self.max_epochs: 
            
            plot_loss(train_losses, self.save_path, val_ious if len(val_ious)>0 else None)

            
            
            
        logger.info('Train time: {} \n'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_train))))
        return train_losses, val_ious
    
    
    def train_epoch(self, e, train_dl):
        self.model.train()
        train_epoch_losses = []
        
        train_classification_loss = []
        train_regression_loss = []
        search_similarity_loss = []
        dynamic_similarity_loss = []
        dissimilarity_loss = []
        
        progress_bar = tqdm(train_dl)
        for batch in progress_bar:
            inputs, targets = self.get_input(batch)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss, loss_dict = self.compute_loss(loss)
            
            train_epoch_losses.append(total_loss.item())
            
            train_classification_loss.append(loss_dict[constants.TARGET_CLASSIFICATION_KEY].item())
            train_regression_loss.append(loss_dict[constants.TARGET_REGRESSION_LABEL_KEY].item())
            search_similarity_loss.append(loss_dict[constants.SIMSIAM_SEARCH_OUT_KEY].item())
            dynamic_similarity_loss.append(loss_dict[constants.SIMSIAM_DYNAMIC_OUT_KEY].item())
            dissimilarity_loss.append(loss_dict[constants.SIMSIAM_NEGATIVE_OUT_KEY].item())
            
            total_loss.backward()
            self.optimizer.step() 
            progress_bar.set_description(f'Training - Epoch {e}/{self.max_epochs} | loss {mean(train_epoch_losses):.3f} | cl_l {mean(train_classification_loss):.3f} | reg_l {mean(train_regression_loss):.3f} | s_sim_l {mean(search_similarity_loss):.3f} | d_sim_l {mean(dynamic_similarity_loss):.3f} | dissim_l {mean(dissimilarity_loss):.3f}')
        
        return mean(train_epoch_losses), mean(train_classification_loss), mean(train_regression_loss), mean(search_similarity_loss), mean(dynamic_similarity_loss), mean(dissimilarity_loss)
    
    def validate_network(self, data_loader, threshold:int = 0.5):
        self.model.eval()
                
        _iou_threshold = 0.01
        # max_samples = self.config.get("max_val_samples", 200)
        seq_ious = {}
        
        
        with inference_mode(): # no_grad():
            progress_bar = tqdm(data_loader)
            for batch in progress_bar:
                for image_files, annotations, dataset_name in batch:
                    if dataset_name not in seq_ious.keys():
                        seq_ious[dataset_name]=[]
                    image_t_0 = read_img(image_files[0])
                    self.tracker.initialize(image_t_0, list(map(int, annotations[0])))
                    num_samples = min(200, len(annotations)) if dataset_name=='lasot' or dataset_name=='nfs'  or dataset_name=='trackingnet' else len(annotations) #min(500, len(annotations))
                    # print('num_samples: ', num_samples)
                    ious = []
                    failure_map = []
                    for i in range(1, num_samples):
                        search_image = read_img(image_files[i])
                        bbox, cls_score = self.tracker.update(search=search_image)
                        iou = get_iou(np.array(bbox), np.array(list(map(int, annotations[i]))))
                        ious.append(iou)
                        failure_map.append(int(iou < _iou_threshold))
                        
                    mean_iou = np.mean(ious)
                    seq_ious[dataset_name].append(mean_iou)
                    
                    iou_str = 'Testing the dataset with the gt (IoU):'
                    for key in seq_ious.keys(): iou_str+=f' | {key}={mean(seq_ious[key]):.3f}'   
                    progress_bar.set_description(iou_str)

        for key in seq_ious.keys():
            seq_ious[key] = mean(seq_ious[key])
            
        return seq_ious
    
    def compute_loss(self, loss: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Return tuple of loss tensor and dictionary of named losses as second argument (if possible)
        """
        total_loss = 0
        for k, v in loss.items():
            total_loss = total_loss + v

        return total_loss, loss

    def get_input(self, data: Dict[str, Any]) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        input_keys = [
            constants.TRACKER_TARGET_TEMPLATE_IMAGE_KEY,
            constants.TRACKER_TARGET_DYNAMIC_TEMPLATE_IMAGE_KEY,
            constants.TRACKER_TARGET_SEARCH_IMAGE_KEY,
            constants.TRACKER_TARGET_DYNAMIC_SEARCH_IMAGE_KEY,
            constants.TRACKER_TARGET_GAUSSIAN_MOVING_MAP,
            constants.TRACKER_TARGET_NEGATIVE_IMAGE_KEY, # currently dont have it
        ]
        target_keys = [
            constants.TARGET_CLASSIFICATION_KEY,
            constants.TARGET_REGRESSION_LABEL_KEY,
            constants.TARGET_REGRESSION_WEIGHT_KEY,
            constants.TRACKER_TARGET_BBOX_KEY,
            constants.TARGET_VISIBILITY_KEY,
           
        ]
        inputs_dict = self._convert_inputs(data, input_keys)
        targets_dict = self._convert_inputs(data, target_keys)
        inputs = [
            inputs_dict[constants.TRACKER_TARGET_TEMPLATE_IMAGE_KEY], 
            inputs_dict[constants.TRACKER_TARGET_DYNAMIC_TEMPLATE_IMAGE_KEY], 
            inputs_dict[constants.TRACKER_TARGET_SEARCH_IMAGE_KEY],
            inputs_dict[constants.TRACKER_TARGET_DYNAMIC_SEARCH_IMAGE_KEY]
            ]
        if constants.TRACKER_TARGET_NEGATIVE_IMAGE_KEY in inputs_dict:
            inputs.append(inputs_dict[constants.TRACKER_TARGET_NEGATIVE_IMAGE_KEY])
        return tuple(inputs), targets_dict

    def _convert_inputs(self, data: Dict[str, Any], keys: Iterable) -> Dict[str, Any]:
        returned = dict()
        for key in keys:
            if key not in data:
                continue
            gt_map = data[key]
            if isinstance(gt_map, np.ndarray):
                gt_map = torch.from_numpy(gt_map)
            if isinstance(gt_map, list):
                gt_map = [item.to(dtype=self.input_type, device=self.device_id, non_blocking=True) for item in gt_map]
            else:
                gt_map = gt_map.to(dtype=self.input_type, device=self.device_id, non_blocking=True)
            returned[key] = gt_map
        return returned
    
    
    def save_network_checkpoint(self, epoch):
        save_dict = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }
        
        save(save_dict, os.path.join(self.save_path, f'trained_model_ckpt_{epoch}.pt'))
        