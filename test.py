import os
import cv2
import imageio.v3 as iio
import warnings
from typing import Dict, Any
import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.utils.utils import _decode_image, get_iou, plot_loss
from core.train import get_tracking_test_datasets
from core.train.train_val import get_collate_for_dataset
from core.utils import prepare_experiment, create_logger
from core.utils.torch_stuff import load_from_lighting

from AEVT_tracker import AEVTTracker
logger = create_logger(__name__)
warnings.filterwarnings("ignore")



def draw_bbox(image, bbox, width: int = 5) -> np.ndarray:
        image = image.copy()
        x, y, w, h = bbox
        return cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), width)


def visualize(frames, tracked_bboxes):
    visualized_frames = []
    for frame, bbox in zip(frames, tracked_bboxes):
        frame = (frame.numpy().transpose(1,2,0)*255).astype('uint8')
        visualized_frames.append(draw_bbox(frame, bbox))
    return visualized_frames


def test_network(tracker, data_loader, save_path, threshold:int = 0.5):

    for batch in tqdm(data_loader, desc='Testing the dataset'):
        for image_files, annotations, dataset_name in batch:
                        
            image_t_0 = _decode_image(image_files[0])
            tracker.initialize(image_t_0, list(map(int, annotations[0])))
            num_samples =len(image_files[0])
            video = [image_t_0]
            tracked_bboxes = [search_image]
            dynamic_image = image_t_0
            prev_dynamic_image = image_t_0
            for i in range(1, num_samples):
                search_image = _decode_image(image_files[i])
                video.append(search_image)
                bbox, cls_score = tracker.update(search=search_image, dynamic=dynamic_image, prev_dynamic=prev_dynamic_image)
                tracked_bboxes.append(bbox)
                if cls_score > threshold:
                    prev_dynamic_image=dynamic_image
                    dynamic_image=search_image

            visualized_video = visualize(video, tracked_bboxes)
            if os.path.exists(os.path.join(save_path,dataset_name))==False: os.makedirs(os.path.join(save_path,dataset_name)) 
            iio.imwrite(os.path.join(save_path,dataset_name, image_files[0].split(os.sep)[-2]), visualized_video, fps=30)



def test(config: Dict[str, Any], save_path, weights_path) -> None:
        
    model = instantiate(config["model"])
    model = load_from_lighting(model, weights_path, map_location=0).cuda().eval()
    tracker: AEVTTracker = instantiate(config["tracker"], model=model)

    test_dataset = get_tracking_test_datasets(config)
    assert test_dataset is not None, "Test Dataset - None"

    collate_fn = get_collate_for_dataset(test_dataset)
    test_dl,_ = DataLoader(
                dataset=test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                collate_fn=collate_fn
            )
    
    test_network(tracker, test_dl, save_path)


@hydra.main(config_name="AEVT_tracker", config_path="core/config")
def run_experiment(hydra_config: DictConfig) -> None:
    config = prepare_experiment(hydra_config)
    logger.info("Experiment dir %s" % config["experiment"]["folder"])
    
    save_path = os.path.join(config["experiment"]["folder"], config["experiment"]["name"])
    if os.path.exists(save_path) == False: os.makedirs(save_path)

    weights_path = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/code/experiments/2023-09-06-16-06-47_Tracking_AEVT/AEVT/trained_model_ckpt_2.pt'
    test(config, save_path, weights_path)
            

# if __name__ == "__main__":
#     run_experiment()