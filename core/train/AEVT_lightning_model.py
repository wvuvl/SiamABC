from typing import Dict, Tuple, Any, List, Union, Callable, Iterable

import cv2
import numpy as np
import torch
from hydra.utils import instantiate
from pytorch_toolbelt.utils.torch_utils import rgb_image_from_tensor
from pytorch_toolbelt.utils.visualization import hstack_autopad, vstack_autopad, vstack_header
from torch import Tensor
from torch.utils.data import ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import DummyLogger
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data.dataloader import default_collate
from torchmetrics import MetricCollection
from torchvision.ops import box_convert, box_iou


from metrics import DatasetAwareMetric, BoxIoUMetric, TrackingFailureRateMetric, box_iou_metric
from models.loss import AEVTLoss
from utils.box_coder import TrackerDecodeResult, AEVTBoxCoder
from utils.utils import read_img, get_iou
from utils.logger import create_logger
import core.constants as constants


####
####
####
# TODO: make it work for dynamic search and gaussian map
####
####
####


logger = create_logger(__name__)


def get_collate_for_dataset(dataset: Union[Dataset, ConcatDataset]) -> Callable:
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


class BaseLightningModel(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], train: Dataset, val: Dataset) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.use_ddp = self.config.get("accelerator", None) == "ddp"
        self.epoch_num = 0
        self.tensorboard_logger = None

    @property
    def is_master(self) -> bool:
        """
        Returns True if the caller is the master node (Either code is running on 1 GPU or current rank is 0)
        """
        return (self.use_ddp is False) or (torch.distributed.get_rank() == 0)

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=self.config.get("metric_mode", "min"),
                                                               factor=0.5,
                                                               patience=5,
                                                               min_lr=1e-6)
        scheduler_config = {"scheduler": scheduler, "monitor": self.config.get("metric_to_monitor", "valid/loss")}
        return [optimizer], [scheduler_config]

    def train_dataloader(self) -> DataLoader:
        """_summary_

        Returns:
            DataLoader: _description_
        """        
        return self._get_dataloader(self.train_dataset, self.config, "train")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dataset, self.config, "val")

    def on_epoch_end(self) -> None:
        self.epoch_num += 1

    def on_pretrain_routine_start(self) -> None:
        if not isinstance(self.logger, DummyLogger):
            for logger in self.logger:
                if isinstance(logger, TensorBoardLogger):
                    self.tensorboard_logger = logger

    def _get_dataloader(self, dataset: Dataset, config: Dict[str, Any], loader_name: str) -> DataLoader:
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

        dataset_config = config[loader_name]
        if "sampler" not in dataset_config or dataset_config["sampler"] == "none":
            sampler = None
        else:
            sampler = self._build_sampler(dataset_config, dataset)

        drop_last = loader_name == "train"

        if self.use_ddp:
            world_size = torch.distributed.get_world_size()
            local_rank = torch.distributed.get_rank()
            sampler = DistributedSampler(dataset, world_size, local_rank)

        should_shuffle = (sampler is None) and (loader_name == "train")
        batch_size = self._get_batch_size(loader_name)
        # Number of workers must not exceed batch size
        num_workers = min(batch_size, self.config["num_workers"])
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )
        return loader

    def _get_batch_size(self, mode: str = "train") -> int:
        if isinstance(self.config["batch_size"], dict):
            return self.config["batch_size"][mode]
        return self.config["batch_size"]


class AEVTLightningModel(BaseLightningModel):
    input_type = torch.float32
    target_type = torch.float32

    def __init__(self, model, config, train, val) -> None:
        super().__init__(model, config, train, val)
        self.tracker = instantiate(config["tracker"], model=self.model)
        self.box_coder = AEVTBoxCoder(tracker_config=config["tracker"])
        self.metrics = MetricCollection(
            {
                "box_iou": BoxIoUMetric(compute_on_step=True),
                "failure_rate": TrackingFailureRateMetric(compute_on_step=True),
            }
        )
        self.dataset_aware_metric = DatasetAwareMetric(metric_name="box_iou", metric_fn=box_iou_metric)
        self.criterion = AEVTLoss(coeffs=config["loss"]["coeffs"])

    def training_step(self, batch: Dict[str, Any], batch_nb: int):
        loss, outputs = self._training_step(batch=batch, batch_nb=batch_nb)
        return loss

    def _training_step(self, batch: Dict[str, Any], batch_nb: int):
        inputs, targets = self.get_input(batch)
        outputs = self.model.forward(inputs)
        loss = self.criterion(outputs, targets)
        total_loss, loss_dict = self.compute_loss(loss)

        decoded_info: TrackerDecodeResult = self.box_coder.decode(
            classification_map=outputs[constants.TARGET_CLASSIFICATION_KEY],
            regression_map=outputs[constants.TARGET_REGRESSION_LABEL_KEY],
        )
        pred_boxes = box_convert(decoded_info.bbox, "xywh", "xyxy")
        gt_boxes = box_convert(targets[constants.TRACKER_TARGET_BBOX_KEY], "xywh", "xyxy")
        visibility_mask = (targets[constants.TARGET_VISIBILITY_KEY][:, 0] == 1).tolist()
        datasets = list(np.array(batch[constants.DATASET_NAME_KEY])[visibility_mask])
        pred_boxes = pred_boxes[visibility_mask]
        gt_boxes = gt_boxes[visibility_mask]
        ious = box_iou(pred_boxes, gt_boxes)

        metrics = self.metrics(ious)

        for metric_name, metric_value in metrics.items():
            self.log(
                f"train/metrics/{metric_name}",
                metric_value,
                on_epoch=True,
            )

        self.dataset_aware_metric.update(mode="train", datasets=datasets, outputs=ious)
        self.log(f"train/loss", total_loss, prog_bar=True, sync_dist=self.use_ddp)
        for key, loss in loss_dict.items():
            self.log(f"train/{key}_loss", loss, sync_dist=self.use_ddp)
        return {"loss": total_loss}, outputs

    def validation_step(self, batch: Tuple[Any, Any, str], batch_nb: int, threshold:int = 0.5) -> Dict[str, Any]:
        _iou_threshold = 0.01
        max_samples = self.config.get("max_val_samples", 200)
        seq_ious = []
        for image_files, annotations, dataset_name in batch:
            image_t_0 = read_img(image_files[0])
            self.tracker.initialize(image_t_0, list(map(int, annotations[0])))
            num_samples = min(max_samples, len(annotations))
            ious = []
            failure_map = []
            dynamic_image = image_t_0
            prev_dynamic_image = image_t_0
            for i in range(1, num_samples):
                search_image = read_img(image_files[i])
                bbox, cls_score = self.tracker.update(search=search_image, dynamic=dynamic_image, prev_dynamic=prev_dynamic_image)
                iou = get_iou(np.array(bbox), np.array(list(map(int, annotations[i]))))
                ious.append(iou)
                failure_map.append(int(iou < _iou_threshold))
                
                # TODO: check if this threholding is proper
                # updating dynamic templates
                if cls_score > threshold:
                    prev_dynamic_image=dynamic_image
                    dynamic_image=search_image

            mean_iou = np.mean(ious)
            self.log(
                "valid/metrics/box_iou",
                mean_iou,
                on_epoch=True,
            )
            self.log(
                f"valid/metrics/{dataset_name}_box_iou",
                mean_iou,
                on_epoch=True,
            )
            self.log(
                f"valid/metrics/{dataset_name}_failure_rate",
                np.mean(failure_map),
                on_epoch=True,
            )
            seq_ious.append(mean_iou)
        return {"box_iou": np.mean(seq_ious)}

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
            constants.TRACKER_TARGET_SEARCH_IMAGE_KEY,
            constants.TRACKER_TARGET_DYNAMIC_IMAGE_KEY,
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
            inputs_dict[constants.TRACKER_TARGET_SEARCH_IMAGE_KEY],
            inputs_dict[constants.TRACKER_TARGET_DYNAMIC_IMAGE_KEY], 
            inputs_dict[constants.TRACKER_TARGET_GAUSSIAN_MOVING_MAP],
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
                gt_map = [item.to(dtype=self.input_type, device=self.device, non_blocking=True) for item in gt_map]
            else:
                gt_map = gt_map.to(dtype=self.input_type, device=self.device, non_blocking=True)
            returned[key] = gt_map
        return returned

    def on_train_epoch_start(self) -> None:
        self.dataset_aware_metric.reset("train")
        super().on_train_epoch_start()
        self.box_coder.to_device(self.device)

    def on_validation_epoch_start(self) -> None:
        self.dataset_aware_metric.reset("valid")
        super().on_validation_epoch_start()
        self.box_coder.to_device(self.device)

    def on_train_epoch_end(self, outputs: Any) -> None:
        metrics = self.dataset_aware_metric.compute("train")
        for metric_name, metric_value in metrics.items():
            self.log(
                f"train/metrics/{metric_name}",
                metric_value,
                on_epoch=True,
            )
        super().on_train_epoch_end(outputs)

    def on_validation_epoch_end(self) -> None:
        metrics = self.dataset_aware_metric.compute("valid")
        for metric_name, metric_value in metrics.items():
            self.log(
                f"valid/metrics/{metric_name}",
                metric_value,
                on_epoch=True,
            )
        self.update_offset()
        self.resample_datasets()
        super().on_validation_epoch_end()

    def on_pretrain_routine_start(self) -> None:
        super().on_pretrain_routine_start()
        self.box_coder.to_device(self.device)
        self.tracker.to_device(self.device)

    def on_train_start(self) -> None:
        super().on_train_start()
        self.box_coder.to_device(self.device)
        self.tracker.to_device(self.device)

    def _denormalize_img(self, input: np.ndarray, idx: int) -> np.ndarray:
        return rgb_image_from_tensor(input[idx])

    def get_visuals(
        self,
        inputs: Dict[str, Tensor],
        outputs: Dict[str, Any],
        score: float,
        max_images=None,
    ) -> np.ndarray:
        decoded_results = self.box_coder.decode(
            regression_map=outputs[constants.TARGET_REGRESSION_LABEL_KEY],
            classification_map=outputs[constants.TARGET_CLASSIFICATION_KEY],
        )
        template_imgs, search_imgs = inputs[constants.TRACKER_TARGET_TEMPLATE_IMAGE_KEY], inputs[constants.TRACKER_TARGET_SEARCH_IMAGE_KEY]
        template_filenames, search_filenames = (
            inputs[constants.TRACKER_TARGET_TEMPLATE_FILENAME_KEY],
            inputs[constants.TRACKER_TARGET_SEARCH_FILENAME_KEY],
        )
        num_images = len(template_imgs)
        if max_images is not None:
            num_images = min(num_images, max_images)

        batch_images = []
        for idx in range(num_images):
            template_img = rgb_image_from_tensor(template_imgs[idx][:3]).copy()
            search_img = rgb_image_from_tensor(search_imgs[idx][:3]).copy()
            pred_x, pred_y, pred_w, pred_h = map(int, decoded_results.bbox.cpu().tolist()[idx])
            gt_x, gt_y, gt_w, gt_h = map(int, inputs[constants.TRACKER_TARGET_BBOX_KEY][idx].cpu().tolist())
            gt_color = (0, 0, 250) if inputs[constants.TARGET_VISIBILITY_KEY][idx].item() == 0.0 else (250, 0, 0)
            search_img = cv2.rectangle(search_img, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h), (0, 250, 0), 2)
            search_img = cv2.rectangle(search_img, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), gt_color, 2)
            img = hstack_autopad(
                [
                    template_img,
                    search_img,
                ]
            )
            img = vstack_header(img, f"S: {inputs[constants.DATASET_NAME_KEY][idx]}, {search_filenames[idx]}")
            img = vstack_header(img, f"T: {inputs[constants.DATASET_NAME_KEY][idx]}, {template_filenames[idx]}")
            batch_images.append(img)

        res_img = vstack_autopad(batch_images)
        res_img = vstack_header(res_img, f"Batch Score {score:.4f}")
        return res_img

    def resample_datasets(self):
        train_dataset = self.train_dataloader().dataset
        datasets_to_update = train_dataset.datasets if type(train_dataset) is ConcatDataset else [train_dataset]
        for dataset in datasets_to_update:
            dataset.resample()

    def update_offset(self):
        if "dynamic_frame_offset" not in self.config:
            return
        params = self.config["dynamic_frame_offset"]
        start_epoch = params["start_epoch"]
        freq = params["freq"]
        step = params["step"]
        max_value = params["max_value"]

        train_dataset = self.train_dataloader().dataset
        datasets_to_update = train_dataset.datasets if type(train_dataset) is ConcatDataset else [train_dataset]
        if (self.current_epoch + 1) >= start_epoch and (self.current_epoch + 1) % freq == 0:
            for dataset in datasets_to_update:
                frame_offset = dataset.item_sampler.frame_offset
                updated_frame_offset = min(max_value, frame_offset + step)
                dataset.item_sampler.frame_offset = updated_frame_offset
                logger.info(
                    f"{dataset.config['root']} frame_offset updated from {frame_offset} to {updated_frame_offset}"
                )
