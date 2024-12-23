
import os
import cv2
import imageio.v3 as iio
import numpy as np
import torch.nn as nn
import torch
from fire import Fire
from tqdm import tqdm
from hydra.utils import instantiate
from typing import List, Optional, Union
from pytorch_toolbelt.utils import transfer_weights
from SiamABC_tracker import SiamABCTracker
from core.utils.hydra import load_hydra_config_from_path
from core.models.custom_bn import replace_layers

def load_model(
    model: nn.Module, checkpoint_path: str, map_location: Optional[Union[int, str]] = None, strict: bool = True
) -> nn.Module:
    map_location = f"cuda:{map_location}" if type(map_location) is int else map_location
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = {
        k.lstrip("module").lstrip("."): v for k, v in checkpoint.items() if k.startswith("module.")
    }

    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        transfer_weights(model, state_dict)
    return model


def get_tracker(config, weights_path: str, lambda_tta: int = 0.1) -> SiamABCTracker:
    
    model = instantiate(config["model"])
        
    replace_layers(model.connect_model.cls_dw, lambda_tta, False)
    replace_layers(model.connect_model.reg_dw,  lambda_tta, False)
    replace_layers(model.connect_model.bbox_tower,  lambda_tta, False)
    replace_layers(model.connect_model.cls_tower,  lambda_tta, False)
    print(model)
    
    model = load_model(model, weights_path, strict=False).cuda().eval()
    tracker: SiamABCTracker = instantiate(config["tracker"], model=model)
    return tracker


def track(tracker: SiamABCTracker, frames: List[np.ndarray], initial_bbox: np.ndarray) -> List[np.ndarray]:
    tracked_bboxes = [initial_bbox]
    tracker.initialize(frames[0], initial_bbox)
            
    for idx, frame in tqdm(enumerate(frames[1:])):
        tracked_bbox,cls_score = tracker.update(frame)
        tracked_bboxes.append(tracked_bbox)
        
    return tracked_bboxes


def draw_bbox(image: np.ndarray, bbox: np.ndarray, width: int = 5) -> np.ndarray:
    image = image.copy()
    x, y, w, h = bbox
    return cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), width)


def visualize(frames: List[np.ndarray], tracked_bboxes: List[np.ndarray]):
    visualized_frames = []
    for frame, bbox in zip(frames, tracked_bboxes):
        visualized_frames.append(draw_bbox(frame, bbox))
    return visualized_frames

import os
def main(
    initial_bbox: List[int] = [416, 414, 61, 97],
    video_path: str = "assets/penguin_in_fog.mp4",
    output_path: str = "outputs/penguin_in_fog.mp4",
    config_path: str = "core/config",
    config_name: str = "SiamABC_tracker",
    model_size: str = "S_Tiny",
    weights_path: str = "assets/S_Tiny/model_S_Tiny_v1.pt",
):
    config = load_hydra_config_from_path(config_path=config_path, config_name=config_name)
    config["model"]["model_size"] = 'S' if model_size=="S_Tiny" else 'M'
        
    tracker = get_tracker(config=config, weights_path=weights_path)
    video, metadata = iio.imread(video_path), iio.immeta(video_path, exclude_applied=False)

    initial_bbox = np.array(initial_bbox).astype(int)
    tracked_bboxes = track(tracker, video, initial_bbox)

    visualized_video = visualize(video, tracked_bboxes)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    iio.imwrite(output_path, visualized_video, fps=metadata["fps"])
    
    head, tail = os.path.split(output_path)
    bbox_dir = os.path.join(head,'bboxes')
    if os.path.exists(bbox_dir) == False: os.makedirs(bbox_dir)
    
    with open(os.path.join(bbox_dir,os.path.splitext(tail)[0]+'.txt'), 'w', encoding='utf-8') as f:
        for i in tracked_bboxes:
            f.write(f'{i[0]} {i[1]} {i[2]} {i[3]} \n')
        
        
if __name__ == '__main__':
    Fire(main)
    
