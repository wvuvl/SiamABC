import os
import random
from abc import ABC
from typing import Any, Dict, Optional, Tuple


import numpy as np
from hydra.utils import instantiate
from got10k.datasets import VOT, GOT10k, NfS
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms

from core.train.preprocessing import torch_resize
from core.utils.gaussian_map import gaussian_label_function
from core.utils.box_coder import AEVTBoxCoder
from core.utils.utils import handle_empty_bbox, _decode_image, ensure_bbox_boundaries, get_extended_image_crop_torch, get_bbox_from_crop_bbox, scale_bbox, \
    get_regression_weight_label, extend_bbox, convert_xywh_to_xyxy, augment_bbox_crop, convert_center_to_bbox, appply_bbox_crop_aug_to_bbox, affine_crop_torch
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
        self.box_coder = AEVTBoxCoder(config["tracker"])
        self.template_resize = torch_resize(self.sizes_config["template_image_size"])
        
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
        
        template_image = _decode_image(template_item["img_path"])
        
        if search_item["img_path"] == template_item["img_path"]:
            search_image = template_image.clone() 
        else:
            search_image = _decode_image(search_item["img_path"]) 
        
        if dynamic_item["img_path"] == template_item["img_path"]:
            dynamic_image = template_image.clone()
        elif dynamic_item["img_path"] == search_item["img_path"]:
            dynamic_image = search_image.clone()
        else:     
            dynamic_image = _decode_image(dynamic_item["img_path"])       
        

        template_bbox = ensure_bbox_boundaries(eval(template_item["bbox"]), img_shape=template_image.shape[1:])
        search_bbox = ensure_bbox_boundaries(eval(search_item["bbox"]), img_shape=search_image.shape[1:])
        
        dynamic_bbox = ensure_bbox_boundaries(eval(dynamic_item["bbox"]), img_shape=dynamic_image.shape[1:])
        
        prev_dynamic_image_shape = eval(prev_dynamic_item['frame_shape'])
        prev_dynamic_bbox = ensure_bbox_boundaries(eval(prev_dynamic_item["bbox"]), img_shape=[prev_dynamic_image_shape[1], prev_dynamic_image_shape[0]])
        
        
        return dict(
            template_image=template_image,
            template_bbox=template_bbox,
            search_image=search_image,
            search_bbox=search_bbox,
            search_presence=search_item["presence"],
            dynamic_image=dynamic_image,
            dynamic_bbox=dynamic_bbox,
            prev_dynamic_bbox=prev_dynamic_bbox,
            prev_dynamic_image_shape=prev_dynamic_image_shape
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
        
        template_crop, template_bbox, search_crop, search_bbox, dynamic_crop, dynamic_bbox, prev_dynamic_bbox = self._get_crops(item_data)
        
        
        crop_size = self.sizes_config["search_image_size"]
        # search_bbox = ensure_bbox_boundaries(np.array(search_bbox), img_shape=(crop_size, crop_size))
        # dynamic_bbox = ensure_bbox_boundaries(np.array(dynamic_bbox), img_shape=(crop_size, crop_size))
        # prev_dynamic_bbox = ensure_bbox_boundaries(np.array(prev_dynamic_bbox), img_shape=(crop_size, crop_size))

        grid_size = self.config["regression_weight_label_size"]
        if search_presence:
            regression_weight_label = get_regression_weight_label(search_bbox, crop_size, grid_size)
            encoded_result = self.box_coder.encode(torch.tensor(search_bbox).reshape(1, 4))
            regression_map = encoded_result.regression_map[0]
            classification_label = encoded_result.classification_label[0]
        else:
            regression_weight_label = torch.zeros(grid_size, grid_size)
            regression_map = torch.zeros(4, grid_size, grid_size)
            classification_label = torch.zeros(1, grid_size, grid_size)
            
        
        dynamic_gaussian_label = gaussian_label_function(torch.tensor(convert_xywh_to_xyxy(dynamic_bbox)).view(1,-1), feat_sz=grid_size, image_sz=crop_size)
        prev_dynamic_gaussian_label = gaussian_label_function(torch.tensor(convert_xywh_to_xyxy(prev_dynamic_bbox)).view(1,-1), feat_sz=grid_size, image_sz=crop_size)
        gaussian_moving_map = torch.concat([prev_dynamic_gaussian_label, dynamic_gaussian_label], dim=0).float()

        
        template_crop = self.template_resize(template_crop)
        # search_crop = search_augmentation(search_crop)
        # dynamic_crop = search_augmentation(dynamic_crop)
        # search_crop = self.torch_resize_search(search_crop)
        # dynamic_crop = self.torch_resize_search(dynamic_crop)
        
        return {
            constants.TARGET_REGRESSION_LABEL_KEY: regression_map,
            constants.TARGET_CLASSIFICATION_KEY: classification_label,
            constants.TARGET_REGRESSION_WEIGHT_KEY: regression_weight_label,
            constants.TRACKER_TARGET_TEMPLATE_IMAGE_KEY: template_crop,
            constants.TRACKER_TEMPLATE_BBOX_KEY: torch.tensor(template_bbox),
            constants.TRACKER_TARGET_SEARCH_IMAGE_KEY: search_crop,
            constants.TRACKER_TARGET_BBOX_KEY: torch.tensor(search_bbox),
            constants.TARGET_VISIBILITY_KEY: np.expand_dims(search_presence, axis=0),
            constants.TRACKER_TARGET_DYNAMIC_IMAGE_KEY: dynamic_crop,
            constants.TRACKER_TARGET_GAUSSIAN_MOVING_MAP: gaussian_moving_map
            
        }
    def get_bbox_from_context(self, bbox, size, context: np.array = None) -> np.array:
            
        bbox, context = get_bbox_from_crop_bbox(
            bbox=bbox,
            context=context)
        
        bbox = handle_empty_bbox(
            ensure_bbox_boundaries(
                np.array(bbox),
                img_shape=(size, size),
            )
        )
        return bbox
    
    def get_search_transform(self, image, bbox, context = None):
        search_size = self.sizes_config["search_image_size"]

        crop, context = get_extended_image_crop_torch(
            image=image,
            context=context
        )
       
        padded_bbox, context  = get_bbox_from_crop_bbox(bbox, context)
       
        crop_bbox = convert_center_to_bbox(
            [
                crop.shape[2] // 2,
                crop.shape[1] // 2,
                crop.shape[2],
                crop.shape[1],
            ],
        )

        modified_context = augment_bbox_crop(
            crop_bbox=crop_bbox, 
            bbox=padded_bbox, image_height=image.shape[1],
            image_width=image.shape[2], 
            scale=self.sizes_config["search_image_scale"],
            shift=self.sizes_config["search_image_shift"]
            )

        new_bbox = appply_bbox_crop_aug_to_bbox(padded_bbox, modified_context, search_size)
        new_crop = affine_crop_torch(crop, modified_context,search_size)

        new_bbox = handle_empty_bbox(
            ensure_bbox_boundaries(
                np.array(new_bbox),
                img_shape=(search_size, search_size),
            )
        )

        return new_crop, new_bbox

    def get_template_transform(self, image, bbox):
        template_size = self.sizes_config["template_image_size"]
        context = extend_bbox(bbox, offset=self.sizes_config["template_bbox_offset"], image_width=image.shape[2], image_height=image.shape[1])
        crop, context = get_extended_image_crop_torch(
            image=image,
            context=context
        )
        padded_bbox, context = get_bbox_from_crop_bbox(bbox,context)
        new_bbox = np.array(scale_bbox(bbox=padded_bbox, padded_crop_w=context[2], padded_crop_h=context[3], crop_size=template_size))

        new_bbox = handle_empty_bbox(
            ensure_bbox_boundaries(
                np.array(new_bbox),
                img_shape=(template_size, template_size),
            )
        )

        return crop, new_bbox

    
    def resample(self):
        self.item_sampler.resample()


    def check_validity(self, bbox_window, bbox):
        return bbox[0]>=bbox_window[0] and bbox[1]>=bbox_window[1] and bbox[2]<=bbox_window[2] and bbox[3]<=bbox_window[3]
    
    
    def _get_crops(self, item_data):
        
        template_crop, template_bbox = self.get_template_transform(
            item_data["template_image"], item_data["template_bbox"], 
        ) 

        context_factor = 2.0
        while True:
            offset = (random.random() * 0.5) + context_factor
            
            dynamic_context = extend_bbox(item_data["dynamic_bbox"], image_width=item_data["dynamic_image"].shape[2], image_height=item_data["dynamic_image"].shape[1], offset=offset)        
            if self.check_validity( convert_xywh_to_xyxy(dynamic_context), convert_xywh_to_xyxy(item_data["search_bbox"])) and \
                self.check_validity( convert_xywh_to_xyxy(dynamic_context), convert_xywh_to_xyxy(item_data["prev_dynamic_bbox"])): 
                    break
            else:
                context_factor*=2
            
            if context_factor>64:
                break
                
            
            
        if context_factor > 64:
            print("too much context factor")
            print(context_factor)
        
        
        dynamic_crop, dynamic_bbox = self.get_search_transform(item_data["dynamic_image"], item_data["dynamic_bbox"], context=dynamic_context)
        
        prev_dynamic_context = extend_bbox(item_data["prev_dynamic_bbox"], image_width=item_data["prev_dynamic_image_shape"][0], image_height=item_data["prev_dynamic_image_shape"][1], offset=offset)  if context_factor > 64 else dynamic_context
        prev_dynamic_bbox =  self.get_bbox_from_context(item_data["prev_dynamic_bbox"], self.sizes_config["search_image_size"], context=prev_dynamic_context)
        
        search_context = extend_bbox(item_data["search_bbox"], image_width=item_data["search_image"].shape[2], image_height=item_data["search_image"].shape[1], offset=offset)  if context_factor > 64 else dynamic_context
        search_crop, search_bbox = self.get_search_transform(item_data["search_image"], item_data["search_bbox"], context=search_context)        
        
        return template_crop, template_bbox, search_crop, search_bbox, dynamic_crop, dynamic_bbox,prev_dynamic_bbox

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
    
