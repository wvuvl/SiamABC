import os
import random
from abc import ABC
from typing import Any, Dict, Optional, Tuple


import albumentations as A
import numpy as np
from hydra.utils import instantiate
from pytorch_toolbelt.utils import image_to_tensor
from got10k.datasets import VOT, GOT10k, NfS
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from core.utils.gaussian_map import gaussian_label_function
from core.train.preprocessing import BBoxCropWithOffsets, get_normalize_fn, TRACKING_AUGMENTATIONS, PHOTOMETRIC_AUGMENTATIONS
from core.utils.box_coder import AEVTBoxCoder
from core.utils.utils import handle_empty_bbox, read_img, ensure_bbox_boundaries, convert_center_to_bbox, get_extended_crop, get_regression_weight_label, extend_bbox, convert_xywh_to_xyxy
import core.constants as constants


def dummy_collate(batch: Any) -> Any:
    return batch


class SequenceDatasetWrapper(Dataset):
    _datasets = {
        "nfs": NfS,
        "got10k": GOT10k,
        "vot": VOT,
    }

    def __init__(self, dataset_name: str, dataset: Dataset):
        self.dataset_name = dataset_name
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return self.dataset_name

    def __getitem__(self, index: int):
        image_files, annotations = self.dataset[index]
        return image_files, annotations, self.dataset_name

    def get_collate_fn(self) -> Any:
        return dummy_collate

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        dataset_name = config.pop("name")
        dataset = cls._datasets[dataset_name](**config)
        return cls(dataset_name=dataset_name, dataset=dataset)

        


def collate_fn(batch: Any) -> Any:
    """
    Almost the same as default_collate, but does not collate indexes and filenames (they are kepts as lists)
    """
    skip_keys = [constants.IMAGE_FILENAME_KEY, constants.SAMPLE_INDEX_KEY, constants.DATASET_NAME_KEY]
    excluded_items = [dict((k, v) for k, v in b.items() if k in skip_keys) for b in batch]
    included_items = [dict((k, v) for k, v in b.items() if k not in skip_keys) for b in batch]

    batch_collated: dict = default_collate(included_items)
    for k in skip_keys:
        out = [item[k] for item in excluded_items if k in item]
        if len(out):
            batch_collated[k] = out

    return batch_collated

class TrackingDataset(ABC):
    def __init__(self, config):
        dataset_config = config["dataset"]
        self.sizes_config = dataset_config.pop("sizes")
        self.config = dataset_config
        self.item_sampler = instantiate(dataset_config["sampling"])
        self.item_sampler.parse_samples()
        self.max_deep_supervision_stride: Optional[int] = config.get("max_deep_supervision_stride", None)
        self.search_context = self.sizes_config["search_context"] * 2
        self.common_transforms = PHOTOMETRIC_AUGMENTATIONS
        self.box_coder = AEVTBoxCoder(config["tracker"])
        
    def __str__(self):
        return self.config["sampling"]["data_path"]

    def __len__(self) -> int:
        return len(self.item_sampler)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_anno = self._get_item_anno(idx=idx) #DONE
        item_data = self._parse_anno(item_anno) #DONE
        item_data = self._transform(item_data) #DONE
        item_dict = self._form_anno_dict(item_data)
        item_dict = self._add_index(idx, item_anno, item_dict)
        return item_dict

    def _get_item_anno(self, idx: int) -> Any:
        return self.item_sampler.extract_sample(idx=idx)

    def _parse_anno(self, item_anno: Any) -> Any:
        template_item, search_item, dynamic_item, prev_dynamic_item = item_anno["template"], item_anno["search"], item_anno["dynamic"], item_anno["prev_dynamic"]
        
        template_image = read_img(template_item["img_path"])
        search_image = read_img(search_item["img_path"])
        dynamic_image = read_img(dynamic_item["img_path"])
        prev_dynamic_image = read_img(prev_dynamic_item["img_path"])

        template_bbox = ensure_bbox_boundaries(eval(template_item["bbox"]), img_shape=template_image.shape[:2])
        search_bbox = ensure_bbox_boundaries(eval(search_item["bbox"]), img_shape=search_image.shape[:2])
        
        dynamic_bbox = ensure_bbox_boundaries(eval(dynamic_item["bbox"]), img_shape=dynamic_image.shape[:2])
        prev_dynamic_bbox = ensure_bbox_boundaries(eval(prev_dynamic_item["bbox"]), img_shape=prev_dynamic_image.shape[:2])
        
        
        return dict(
            template_image=template_image,
            template_bbox=template_bbox,
            search_image=search_image,
            search_bbox=search_bbox,
            search_presence=search_item["presence"],
            dynamic_image=dynamic_image,
            dynamic_bbox=dynamic_bbox,
            prev_dynamic_image=prev_dynamic_image,
            prev_dynamic_bbox=prev_dynamic_bbox
        )

    def _form_anno_dict(self, item_data: Any) -> Any:
        return item_data

    def _add_index(self, idx: int, annotation: Any, item_dict: Dict[str, Any]) -> Dict[str, Any]:
        template_item, search_item, dynamic_item, prev_dynamic_item = annotation["template"], annotation["search"], annotation["dynamic"], annotation["prev_dynamic"]
        item_dict.update(
            {   constants.TRACKER_TARGET_PREV_DYNAMIC_FILENAME_KEY: prev_dynamic_item["img_path"],
                constants.TRACKER_TARGET_DYNAMIC_FILENAME_KEY: dynamic_item["img_path"],
                constants.TRACKER_TARGET_SEARCH_FILENAME_KEY: search_item["img_path"],
                constants.TRACKER_TARGET_TEMPLATE_FILENAME_KEY: template_item["img_path"],
                
                constants.TRACKER_TARGET_PREV_DYNAMIC_INDEX_KEY: prev_dynamic_item.name,
                constants.TRACKER_TARGET_DYNAMIC_INDEX_KEY: dynamic_item.name,
                constants.TRACKER_TARGET_SEARCH_INDEX_KEY: search_item.name,
                constants.TRACKER_TARGET_TEMPLATE_INDEX_KEY: template_item.name,
                
                constants.SAMPLE_INDEX_KEY: idx,
                constants.DATASET_NAME_KEY: search_item["dataset"],
            }
        )
        return item_dict

    def _transform(self, item_data: Any) -> Any:
        search_presence = item_data["search_presence"]
        
        template_crop, template_bbox, search_crop, search_bbox, dynamic_crop, dynamic_bbox, prev_dynamic_crop,  prev_dynamic_bbox = self._get_crops(item_data)
        
        
        template_crop, search_crop, dynamic_crop = self._add_color_augs(dynamic_image=dynamic_crop, search_image=search_crop, template_image=template_crop)
        
        template_crop, template_bbox = self.transform(image=template_crop, bbox=template_bbox)
        search_crop, search_bbox = self.transform(image=search_crop, bbox=search_bbox)
        dynamic_crop, dynamic_bbox = self.transform(image=dynamic_crop, bbox=dynamic_bbox)
        # prev_dynamic_crop, prev_dynamic_bbox = self.transform(image=prev_dynamic_crop, bbox=prev_dynamic_bbox) # currently, prev_dynamic_crop does not get fed into the network

        crop_size = self.sizes_config["search_image_size"]
        search_bbox = ensure_bbox_boundaries(np.array(search_bbox), img_shape=(crop_size, crop_size))
        dynamic_bbox = ensure_bbox_boundaries(np.array(dynamic_bbox), img_shape=(crop_size, crop_size))
        prev_dynamic_bbox = ensure_bbox_boundaries(np.array(prev_dynamic_bbox), img_shape=(crop_size, crop_size))

        grid_size = self.config["regression_weight_label_size"]
        if search_presence:
            regression_weight_label = get_regression_weight_label(search_bbox, crop_size, grid_size)
            encoded_result = self.box_coder.encode(torch.from_numpy(search_bbox).reshape(1, 4))
            regression_map = encoded_result.regression_map[0]
            classification_label = encoded_result.classification_label[0]
        else:
            regression_weight_label = torch.zeros(grid_size, grid_size)
            regression_map = torch.zeros(4, grid_size, grid_size)
            classification_label = torch.zeros(1, grid_size, grid_size)
            
        
        dynamic_gaussian_label = gaussian_label_function(torch.tensor(convert_xywh_to_xyxy(dynamic_bbox)).view(1,-1), feat_sz=grid_size, image_sz=crop_size)
        prev_dynamic_gaussian_label = gaussian_label_function(torch.tensor(convert_xywh_to_xyxy(prev_dynamic_bbox)).view(1,-1), feat_sz=grid_size, image_sz=crop_size)
        gaussian_moving_map = torch.concat([prev_dynamic_gaussian_label, dynamic_gaussian_label], dim=0).float()
    
        return {
            constants.TARGET_REGRESSION_LABEL_KEY: regression_map,
            constants.TARGET_CLASSIFICATION_KEY: classification_label,
            constants.TARGET_REGRESSION_WEIGHT_KEY: regression_weight_label,
            constants.TRACKER_TARGET_TEMPLATE_IMAGE_KEY: image_to_tensor(template_crop),
            constants.TRACKER_TEMPLATE_BBOX_KEY: torch.tensor(template_bbox),
            constants.TRACKER_TARGET_SEARCH_IMAGE_KEY: image_to_tensor(search_crop),
            constants.TRACKER_TARGET_BBOX_KEY: torch.tensor(search_bbox),
            constants.TARGET_VISIBILITY_KEY: np.expand_dims(search_presence, axis=0),
            constants.TRACKER_TARGET_DYNAMIC_IMAGE_KEY: image_to_tensor(dynamic_crop),
            constants.TRACKER_TARGET_GAUSSIAN_MOVING_MAP: gaussian_moving_map
            
        }

    def _add_color_augs(self, dynamic_image: np.ndarray, search_image: np.ndarray, template_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        color_aug = A.Compose(TRACKING_AUGMENTATIONS, additional_targets={"search_image": "image", "dynamic_image": "image"})
        aug_res = color_aug(image=template_image, search_image=search_image, dynamic_image=dynamic_image)
        return aug_res["image"], aug_res["search_image"], aug_res["dynamic_image"]
    
    


    def get_search_transform(self, image: np.array, bbox: np.array, context: np.array = None) -> Tuple[np.array, np.array]:
        search_size = self.sizes_config["search_image_size"]
        crop, bbox, context = get_extended_crop(
            image=image,
            bbox=bbox,
            crop_size=search_size,
            context=context
        )
        
        
        bbox_crop = convert_center_to_bbox(
            [
                crop.shape[0] // 2,
                crop.shape[1] // 2,
                search_size,
                search_size,
            ],
        )
        crop_aug = BBoxCropWithOffsets(
            bbox_crop=bbox_crop,
            scale=self.sizes_config["search_image_scale"],
            shift=self.sizes_config["search_image_shift"],
            crop_size=search_size,
            p=0.5
        )
        result = crop_aug(image=crop, bboxes=[bbox])
        crop, bbox = result["image"], result["bboxes"][0]
        bbox = handle_empty_bbox(
            ensure_bbox_boundaries(
                np.array(bbox),
                img_shape=(search_size, search_size),
            )
        )
        return crop, bbox

    def get_template_transform(self, image: np.array, bbox: np.array) -> Tuple[np.array, np.array]:
        template_size = self.sizes_config["template_image_size"]
        context = extend_bbox(bbox, offset=self.sizes_config["template_bbox_offset"], image_width=image.shape[1], image_height=image.shape[0])
        crop, bbox, _ = get_extended_crop(
            image=image,
            bbox=bbox,
            crop_size=template_size,
            context=context
        )
        bbox = handle_empty_bbox(
            ensure_bbox_boundaries(
                np.array(bbox),
                img_shape=(template_size, template_size),
            )
        )
        return crop, bbox

    def resample(self):
        self.item_sampler.resample()

    def _get_search_context(self):
        context_range = 0.5 #self.sizes_config.get("context_range")
        min_context = 1.0 # self.search_context - context_range / 2
        return (random.random() * context_range) + min_context
    
    def transform(self, image, bbox):
        """
        image - crop image with centred bbox with additional context amount
        bbox - centred bounding box inside crop

        Here we add some shift, scale transformations, because we have crop with centred bounding box
        We do it in following steps:
        1) Apply photometric transformations to image
        2) get centred square (crop_size, crop_size) crop bounding box, which contains centred object bounding box
        3) get nearly centred centred crop from centred crop bbox using BBoxCropWithOffsets and apply changes to
        object bounding box and image
        """
        full_aug_list = [*self.common_transforms, get_normalize_fn(self.config.get("normalize", "imagenet"))]
        bbox_params = {"format": "coco", "min_visibility": 0, "label_fields": ["category_id"], "min_area": 0}
        transform = A.Compose(full_aug_list, bbox_params=bbox_params)
        result = transform(image=image, bboxes=[bbox], category_id=["bbox"])
        image, bbox = result["image"], np.array(result["bboxes"][0])
        return image, bbox

    def check_validity(self, bbox_window, bbox):
        return bbox[0]>=bbox_window[0] and bbox[1]>=bbox_window[1] and bbox[2]<=bbox_window[2] and bbox[3]<=bbox_window[3]
        
    def _get_crops(self, item_data):
        template_crop, template_bbox = self.get_template_transform(
            item_data["template_image"], item_data["template_bbox"], 
        )
        
        context_factor = 2.0
        while True:
            offset=(random.random() * 0.5) + context_factor
            dynamic_context = extend_bbox(item_data["dynamic_bbox"], image_width=item_data["dynamic_image"].shape[1], image_height=item_data["dynamic_image"].shape[0], offset=offset)
            
            if self.check_validity( convert_xywh_to_xyxy(dynamic_context), convert_xywh_to_xyxy(item_data["search_bbox"])) and \
                self.check_validity( convert_xywh_to_xyxy(dynamic_context), convert_xywh_to_xyxy(item_data["prev_dynamic_bbox"])): 
                    break
            else:
                # print(f"Object not within the search region, increasing the region by {100*context_factor}%")
                context_factor += 2.0
                
        dynamic_crop, dynamic_bbox = self.get_search_transform(item_data["dynamic_image"], item_data["dynamic_bbox"], context=dynamic_context)
        search_crop, search_bbox = self.get_search_transform(item_data["search_image"], item_data["search_bbox"], context=dynamic_context)
        prev_dynamic_crop,  prev_dynamic_bbox = self.get_search_transform(item_data["prev_dynamic_image"], item_data["prev_dynamic_bbox"], context=dynamic_context)
        return template_crop, template_bbox, search_crop, search_bbox, dynamic_crop, dynamic_bbox, prev_dynamic_crop,  prev_dynamic_bbox

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrackingDataset":
        return cls(config)

    def get_collate_fn(self) -> Any:
        """
        Returns default collate method. If you need custom collate logic - override this
        method and return custom collate function.
        BaseModel will check whether dataset has 'collate_fn' attribute and use it whenever possible.
        """
        return collate_fn
    
