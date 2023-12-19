# from __future__ import absolute_import, print_function
from got10k.trackers import Tracker
from got10k import experiments
from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig
import numpy as np
import os
import torch
from core.utils import prepare_experiment
from AEVT_tracker import AEVTTracker
from core.utils.torch_stuff import load_from_lighting
from eval_data.eval_avist_otb_style import ExperimentAVIST
from eval_data.eval_itb_otb_style import ExperimentITB
from eval_data.eval_trackingnet_otb_style import ExperimentTrackingNet

DTB70_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70'
TCOLOR128_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/Temple-color-128'
ROOT_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k'
LASOT_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/LaSOT/LaSOT'
TRACKINGNET_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/TrackingNet'
VOT_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/vot2018'
VOT_DIR_2019 = '/new_local_storage/zaveri/SOTA_Tracking_datasets/vot2019'
NFS_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/NFS/NFS/NFS'
OTB_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/OTB'
UAV_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/UAV123'
UAV_10_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/UAV123_10fps'
TC128_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/TC128'


AVIST_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/avist/sequences'
AVIST_JSON = '/new_local_storage/zaveri/SOTA_Tracking_datasets/avist/AVIST.json'


ITB_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/ITB'
ITB_JSON =  '/new_local_storage/zaveri/SOTA_Tracking_datasets/ITB/ITB.json'
        
        
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

        # box[0] = box[0]-1
        # box[1] = box[1]-1

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

    weights_path = '/new_local_storage/zaveri/code/experiments/2023-11-01-22-11-46_Tracking_SiamABC_small_full_no_correspondence_loss/AEVT/trained_model_ckpt_15.pt'
    # weights_path = '/new_local_storage/zaveri/code/experiments/2023-10-24-21-24-32_Tracking_SiamABC_small_full/AEVT/trained_model_ckpt_17.pt'
    # weights_path = '/new_local_storage/zaveri/code/experiments/2023-10-26-17-03-15_Tracking_SiamABC_resnet50_full/AEVT/trained_model_ckpt_20.pt'
    # weights_path = '/new_local_storage/zaveri/code/experiments/2023-10-31-18-28-55_Tracking_SiamABC_resnet50_layer_4_full/AEVT/trained_model_ckpt_20.pt'
    # weights_path = '/new_local_storage/zaveri/code/experiments/2023-10-29-21-02-32_Tracking_SiamABC_regmetX_full/AEVT/trained_model_ckpt_15.pt'
    
    # got10k
    # weights_path = '/new_local_storage/zaveri/code/experiments/2023-10-28-16-11-18_Tracking_SiamABC_regnetX_got10k/AEVT/trained_model_ckpt_26.pt'
    # weights_path = '/new_local_storage/zaveri/code/experiments/2023-11-04-22-08-52_Tracking_SiamABC_resnet50_layer_4_L_got10k/AEVT/trained_model_ckpt_22.pt'
    # weights_path = '/new_local_storage/zaveri/code/experiments/2023-10-27-15-40-48_Tracking_SiamABC_resnet50_got10k/AEVT/trained_model_ckpt_23.pt'
    # weights_path = '/new_local_storage/zaveri/code/experiments/2023-10-26-15-01-39_Tracking_SiamABC_small_got10k/AEVT/trained_model_ckpt_22.pt'
    
    config["model"]["model_size"] = 'S'
    config["tracker"]["dynamic_update"] = False
    config["tracker"]["smooth"] = True
    # config["tracker"]["penalty_k"] = 0.153 # 0.055 #0.194 # 0.098
    # config["tracker"]["window_influence"] = 0.223 # 0.448 # 0.385 # 0.181 
    # config["tracker"]["lr"] = 0.605 #0.67 #0.33 #0.348
    
    model = instantiate(config["model"])
    model = load_from_lighting(model, weights_path, map_location=0).cuda().eval()
    aevt_tracker: AEVTTracker = instantiate(config["tracker"], model=model)
    save_path = os.path.join(config["experiment"]["folder"], config["experiment"]["name"])
    if os.path.exists(save_path) == False: os.makedirs(save_path)
    # setup tracker
    tracker = IdentityTracker(aevt_tracker)



    
    # # setup experiment (ITB)
    # experiment = ExperimentITB(
    #     root_dir=ITB_DIR,
    #     json_file=ITB_JSON,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])
    
    
    # experiment = ExperimentTrackingNet(
    #     root_dir=TRACKINGNET_DIR,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])

    # # setup experiment (AVIST)
    # experiment = ExperimentAVIST(
    #     root_dir=AVIST_DIR,
    #     json_file=AVIST_JSON,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])

    # # setup experiment (TColor-128)
    # experiment = experiments.ExperimentTColor128(
    #     root_dir=TCOLOR128_DIR,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])

    # # setup experiment DTB170)
    # experiment = experiments.ExperimentDTB70(
    #     root_dir=DTB70_DIR,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])

#     # NfS
    # experiment = experiments.ExperimentNfS(
    #     root_dir=NFS_DIR,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports'),        # where to store evaluation reports
    #     fps=30
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])

    # # setup experiment (validation subset)
    # experiment = experiments.ExperimentGOT10k(
    #     root_dir=ROOT_DIR,          # GOT-10k's root directory
    #     subset='val',               # 'train' | 'val' | 'test'
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports'),        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False, save_video=False)
    # experiment.report([tracker.name])
    
    experiment = experiments.ExperimentGOT10k(
        root_dir=ROOT_DIR,          # GOT-10k's root directory
        subset='test',               # 'train' | 'val' | 'test'
        result_dir=os.path.join(save_path,'results'),       # where to store tracking results
        report_dir=os.path.join(save_path,'reports'),        # where to store evaluation reports
    )
    experiment.run(tracker, visualize=False, save_video=False)
    experiment.report([tracker.name])





    # # setup experiment (OTB100)
    # experiment = experiments.ExperimentOTB(
    #     root_dir=OTB_DIR,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])



    # # setup experiment (UAV123)
    # experiment = experiments.ExperimentUAV123(
    #     root_dir=UAV_DIR,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])

    # # setup experiment (validation subset)
    # experiment = experiments.ExperimentLaSOT(
    #     root_dir=LASOT_DIR,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])





#     # # setup experiment (validation subset)
#     # experiment = ExperimentVOT(
#     #     root_dir=VOT_DIR,
#     #     version=2018,
#     #     experiments='supervised' ,
#     #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
#     #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
#     # )
#     # experiment.run(tracker, visualize=False)
#     # experiment.report([tracker.name])

#     # # setup experiment (validation subset)
#     # experiment = ExperimentVOT(
#     #     root_dir=VOT_DIR_2019,
#     #     version=2019,
#     #     experiments='supervised' ,
#     #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
#     #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
#     # )
#     # experiment.run(tracker, visualize=False)
#     # experiment.report([tracker.name])


# #     # # NfS
#     experiment =  experiments.ExperimentNfS(
#         root_dir=NFS_DIR,
#         result_dir=os.path.join(save_path,'results'),       # where to store tracking results
#         report_dir=os.path.join(save_path,'reports'),        # where to store evaluation reports
#         fps=240
#     )
#     experiment.run(tracker, visualize=False)
#     experiment.report([tracker.name])

if __name__ == '__main__':

    config = run_experiment()

