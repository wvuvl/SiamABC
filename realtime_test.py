
import os
import cv2
from pathlib import Path
import numpy as np
from hydra.utils import instantiate

from AEVT_tracker import AEVTTracker
from core.utils.torch_stuff import load_from_lighting
from core.utils.hydra import load_hydra_config_from_path
from core.utils.utils import read_img

class loadfromfolder:

    """Helper function to load any video frames without gt"""

    def __init__(self, video_dir):
        """Init folder"""

        self._video_dir = video_dir
        self._videos = {}

    def get_video_frames(self):
        """Get video frames from folder"""

        vid_dir = self._video_dir
        vid_frames = [str(img_path) for img_path in
                      Path(vid_dir).glob('*.jpg')]
        if len(vid_frames) == 0:
            vid_frames = [str(img_path) for img_path in
                          Path(vid_dir).glob('*.png')]
            
        if len(vid_frames) == 0:
            vid_frames = [str(img_path) for img_path in
                          Path(vid_dir).glob('*.tif')]
        list_of_frames = sorted(vid_frames)

        self._vid_frames = [list_of_frames]

        return self._vid_frames
    
    
class Tracker:

    refPt = []
    image = []
    cv2.namedWindow('image')
    def click_and_crop(event, x, y, flags, param):
        # grab references to the global variables
        global refPt, cropping
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            refPt.append((x, y))
            cropping = False
            # draw a rectangle around the region of interest
            global image
            img_dbg = np.copy(image)
            img_dbg = cv2.rectangle(img_dbg, refPt[0], refPt[1], (0, 255, 0), 2)
           
            cv2.imshow("image", cv2.cvtColor(img_dbg, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            
    cv2.setMouseCallback("image", click_and_crop)  

    def __init__(self, 
        frames_path: str,
        config_path: str = "core/config",
        config_name: str = "AEVT_tracker",
        weights_path: str = "/media/ramzaveri/12F9CADD61CB0337/cell_tracking/code/AEVT/models/small/trained_model_ckpt_20.pt",
        ):
        """load model """
        loader = loadfromfolder(frames_path)
        self._vid_frames = loader.get_video_frames()
        
        self.tracker = self.get_tracker(config_path=config_path, config_name=config_name, weights_path=weights_path)

    def get_tracker(self, config_path: str, config_name: str, weights_path: str) -> AEVTTracker:
        config = load_hydra_config_from_path(config_path=config_path, config_name=config_name)
        model = instantiate(config["model"])
        model = load_from_lighting(model, weights_path, map_location=0).cuda().eval()
        tracker: AEVTTracker = instantiate(config["tracker"], model=model)
        return tracker


    
    def track(self):
    
        """Track"""
        vid_frames = self._vid_frames[0]
        num_frames = len(vid_frames)
        f_path = vid_frames[0]
        ext = os.path.splitext(f_path)[-1]
        prev = read_img(f_path)
        
        global image
        image = prev
        while True:
            
            prev_out = np.copy(prev)
            cv2.imshow('image', cv2.cvtColor(prev_out, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                (x1, y1), (x2, y2) = refPt[0], refPt[1]
                bbox_0 =[x1, y1, x2, y2]
                break
            elif key == ord('r'):
                (x1, y1), (x2, y2) = refPt[0], refPt[1]
                bbox_0 = [x1, y1, x2, y2]
                break
        
        tracked_bboxes = [[bbox_0[0],bbox_0[1],bbox_0[2]-bbox_0[0], bbox_0[3]-bbox_0[1]]]
        self.tracker.initialize(image, tracked_bboxes[0])
        dynamic_frame = image
        prev_dynamic_frame = image
    
        for i in range(1, num_frames):
            f_path = vid_frames[i]
            curr =  read_img(f_path)
            
            tracked_bbox,cls_score = self.tracker.update(curr,dynamic_frame,prev_dynamic_frame)
            tracked_bboxes.append(tracked_bbox)            
            bbox = [tracked_bbox[0],tracked_bbox[1], tracked_bbox[2]+tracked_bbox[0], tracked_bbox[3]+tracked_bbox[1]]
            if cls_score > 0.5:
                prev_dynamic_frame=dynamic_frame
                dynamic_frame=curr

            if cv2.waitKey(1) & 0xFF == ord('p'):
                while True:
                    image = curr
                    cv2.imshow("image", cv2.cvtColor(curr, cv2.COLOR_RGB2BGR))
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("s"):
                        (x1, y1), (x2, y2) = refPt[0], refPt[1]
                        bbox_0 = [x1, y1, x2, y2] 
                        break

            curr_dbg = np.copy(curr)
            curr_dbg = cv2.rectangle(curr_dbg, (int(bbox[0]),
                                                int(bbox[1])),
                                    (int(bbox[2]), int(bbox[3])), (255, 255, 0), 2)

            cv2.imshow('image', cv2.cvtColor(curr_dbg, cv2.COLOR_RGB2BGR))
            
            cv2.waitKey(20)


if __name__ == "__main__":
    frames_path = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/got10k/OTB/BlurCar1/img'
    AEVT_tracker = Tracker(frames_path, weights_path='/media/ramzaveri/12F9CADD61CB0337/cell_tracking/code/experiments/2023-08-29-16-47-35_Tracking_AEVT/AEVT/trained_model_ckpt_33.pt')
    AEVT_tracker.track()