
import os
import cv2
import imageio.v3 as iio
import numpy as np
from fire import Fire
from tqdm import tqdm
from hydra.utils import instantiate
from typing import Optional, List

from SiamABC_tracker import SiamABCTracker
from core.utils.torch_stuff import load_from_lighting
from core.utils.hydra import load_hydra_config_from_path

    

def get_tracker(config_path: str, config_name: str, weights_path: str) -> SiamABCTracker:
    config = load_hydra_config_from_path(config_path=config_path, config_name=config_name)
    model = instantiate(config["model"])
    model = load_from_lighting(model, weights_path).cuda().eval()
    tracker: SiamABCTracker = instantiate(config["tracker"], model=model)
    return tracker


def track(tracker: SiamABCTracker, frames: List[np.ndarray], initial_bbox: np.ndarray) -> List[np.ndarray]:
    tracked_bboxes = [initial_bbox]
    tracker.initialize(frames[0], initial_bbox)
    
    bbox = tracked_bboxes[0]
    bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
    img = frames[0].copy()
    frame_w_rec = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 2)
    
    dynamic_frame = frames[0]
    prev_dynamic_frame = frames[0]
    
    for idx, frame in tqdm(enumerate(frames[1:])):
        
        tracked_bbox,cls_score = tracker.update(frame,dynamic_frame,prev_dynamic_frame)
        tracked_bboxes.append(tracked_bbox)
        if cls_score > 0.5:
            prev_dynamic_frame=dynamic_frame
            dynamic_frame=frame

        bbox = tracked_bboxes[idx+1]
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        frame_w_rec = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 2)
        # print()
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
    initial_bbox: List[int] = [489, 188, 497, 140],
    video_path: str = "assets/airplane_in_rain.mp4",
    output_path: str = "outputs/airplane_in_rain.mp4",
    config_path: str = "core/config",
    config_name: str = "SiamABC_tracker",
    weights_path: str = "/media/ramzaveri/12F9CADD61CB0337/cell_tracking/code/experiments/2023-08-22-12-26-28_Tracking_SiamABC/SiamABC/trained_model_ckpt_7.pt",
):
    tracker = get_tracker(config_path=config_path, config_name=config_name, weights_path=weights_path)
    video, metadata = iio.imread(video_path), iio.immeta(video_path, exclude_applied=False)
    # print(metadata)
    initial_bbox = np.array(initial_bbox).astype(int)
    tracked_bboxes = track(tracker, video, initial_bbox)
    # print(initial_bbox)
    # print(tracked_bboxes)
    visualized_video = visualize(video, tracked_bboxes)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # print(visualized_video)
    iio.imwrite(output_path, visualized_video, fps=metadata["fps"])
    
    head, tail = os.path.split(output_path)
    bbox_dir = os.path.join(head,'bboxes')
    if os.path.exists(bbox_dir) == False: os.makedirs(bbox_dir)
    
    with open(os.path.join(bbox_dir,os.path.splitext(tail)[0]+'.txt'), 'w', encoding='utf-8') as f:
        for i in tracked_bboxes:
            f.write(f'{i[0]} {i[1]} {i[2]} {i[3]} \n')
        

if __name__ == '__main__':
    Fire(main)
    
