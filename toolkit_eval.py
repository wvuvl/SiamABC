# from __future__ import absolute_import, print_function
from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k, ExperimentOTB, ExperimentUAV123, ExperimentVOT, ExperimentNfS, ExperimentLaSOT
from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig
import numpy as np
import os
import torch
from core.utils import prepare_experiment
from AEVT_tracker import AEVTTracker
from core.utils.torch_stuff import load_from_lighting

ROOT_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k'
LASOT_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/LaSOT/LaSOT'
TRACKINGNET_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/TrackingNet'
VOT_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/vot2018'
VOT_DIR_2019 = '/new_local_storage/zaveri/SOTA_Tracking_datasets/vot2019'
NFS_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/NFS/NFS/NFS'
OTB_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/OTB'
UAV_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/UAV123'



class IdentityTracker(Tracker):
    """Example on how to define a tracker.

        To define a tracker, simply override ``init`` and ``update`` methods
            from ``Tracker`` with your own pipelines.
    """
    def __init__(self, tracker):
        super(IdentityTracker, self).__init__(
            name='AEVTTracker', # name of the tracker
            is_deterministic=True   # deterministic (True) or stochastic (False)
        )
        self.tracker = tracker
        
       
    @torch.no_grad()
    def init(self, image, box):
        """Initialize your tracking model in the first frame
        
        Arguments:
            image {PIL.Image} -- Image in the first frame.
            box {np.ndarray} -- Target bounding box (4x1,
                [left, top, width, height]) in the first frame.
        """
        image = np.asarray(image)
        self.tracker.initialize(image, box)
        self.box = box

    @torch.no_grad()
    def update(self, image):
        """Locate target in an new frame and return the estimated bounding box.
        
        Arguments:
            image {PIL.Image} -- Image in a new frame.
        
        Returns:
            np.ndarray -- Estimated target bounding box (4x1,
                [left, top, width, height]) in ``image``.
        """
        search_image = np.asarray(image)
        bbox, cls_score = self.tracker.update(search=search_image)
        # self.bbox = bbox.astype('int')
        
        return bbox

@hydra.main(config_name="AEVT_tracker", config_path="core/config")
def run_experiment(hydra_config: DictConfig):
    config = prepare_experiment(hydra_config)
    print("Experiment dir %s" % config["experiment"]["folder"])
    
    weights_path = '/new_local_storage/zaveri/code/experiments/2023-10-21-22-40-46_Tracking_SiamABC_regnetX_full/AEVT/trained_model_ckpt_9.pt'
    model = instantiate(config["model"])
    model = load_from_lighting(model, weights_path, map_location=0).cuda().eval()
    aevt_tracker: AEVTTracker = instantiate(config["tracker"], model=model)
    save_path = os.path.join(config["experiment"]["folder"], config["experiment"]["name"])
    if os.path.exists(save_path) == False: os.makedirs(save_path)
    # setup tracker
    tracker = IdentityTracker(aevt_tracker)
    
    # # NfS
    # experiment = ExperimentNfS(
    #     root_dir=NFS_DIR,            
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports'),        # where to store evaluation reports
    #     fps=30
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])

    # # setup experiment (validation subset)
    # experiment = ExperimentGOT10k(
    #     root_dir=ROOT_DIR,          # GOT-10k's root directory
    #     subset='val',               # 'train' | 'val' | 'test'
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports'),        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False, save_video=False)
    # experiment.report([tracker.name])
    
    
    # experiment = ExperimentGOT10k(
    #     root_dir=ROOT_DIR,          # GOT-10k's root directory
    #     subset='test',               # 'train' | 'val' | 'test'
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports'),        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False, save_video=True)
    # experiment.report([tracker.name])
    
    
    # # setup experiment (validation subset)
    # experiment = ExperimentVOT(
    #     root_dir=VOT_DIR,         
    #     version=2018,
    #     experiments='supervised' ,              
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])
    
    # # setup experiment (validation subset)
    # experiment = ExperimentVOT(
    #     root_dir=VOT_DIR_2019,         
    #     version=2019,
    #     experiments='supervised' ,              
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])
    
    # setup experiment (OTB100)
    experiment = ExperimentOTB(
        root_dir=OTB_DIR,
        result_dir=os.path.join(save_path,'results'),       # where to store tracking results
        report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    )
    experiment.run(tracker, visualize=False)
    experiment.report([tracker.name])
    
    
    # # setup experiment (UAV123)
    # experiment = ExperimentUAV123(
    #     root_dir=UAV_DIR,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])
    
    # # NfS
    # experiment = ExperimentNfS(
    #     root_dir=NFS_DIR,            
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports'),        # where to store evaluation reports
    #     fps=240
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])
    
    
    # # setup experiment (validation subset)
    # experiment = ExperimentLaSOT(
    #     root_dir=LASOT_DIR,               
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])
    
    
if __name__ == '__main__':
    
    config = run_experiment()
    
    