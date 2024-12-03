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
from SiamABC_tracker import SiamABCTracker
from core.utils.torch_stuff import load_from_lighting
from core.models.custom_bn import replace_layers
from eval_data.eval_avist_otb_style import ExperimentAVIST
from eval_data.eval_itb_otb_style import ExperimentITB
from eval_data.eval_trackingnet_otb_style import ExperimentTrackingNet
import argparse
import sys
import multiprocessing
from itertools import product
import time
from datetime import timedelta



norm_lambda = 0.25
contineous = True

DTB70_DIR = '/luna_data/zaveri/SOTA_Tracking_datasets/DTB70'
TCOLOR128_DIR = '/luna_data/zaveri/SOTA_Tracking_datasets/Temple-color-128'
ROOT_DIR = '/luna_data/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k'
LASOT_DIR = '/luna_data/zaveri/SOTA_Tracking_datasets/LaSOT/LaSOT'
TRACKINGNET_DIR = '/luna_data/zaveri/SOTA_Tracking_datasets/TrackingNet'
VOT_DIR = '/luna_data/zaveri/SOTA_Tracking_datasets/vot2018'
VOT_DIR_2019 = '/luna_data/zaveri/SOTA_Tracking_datasets/vot2019'
NFS_DIR = '/luna_data/zaveri/SOTA_Tracking_datasets/NFS/NFS/NFS'
OTB_DIR = '/luna_data/zaveri/SOTA_Tracking_datasets/OTB'
UAV_DIR = '/luna_data/zaveri/SOTA_Tracking_datasets/UAV123'
UAV_10_DIR = '/luna_data/zaveri/SOTA_Tracking_datasets/UAV123_10fps'
TC128_DIR = '/luna_data/zaveri/SOTA_Tracking_datasets/TC128'

AVIST_DIR = '/luna_data/zaveri/SOTA_Tracking_datasets/avist/sequences'
AVIST_JSON = '/luna_data/zaveri/SOTA_Tracking_datasets/avist/AVIST.json'

ITB_DIR = '/luna_data/zaveri/SOTA_Tracking_datasets/ITB'
ITB_JSON = '/luna_data//zaveri/SOTA_Tracking_datasets/ITB/ITB.json'

class IdentityTracker(Tracker):
    """Example on how to define a tracker.

        To define a tracker, simply override ``init`` and ``update`` methods
            from ``Tracker`` with your own pipelines.
    """
    def __init__(self, tracker):
        super(IdentityTracker, self).__init__(
            name='SiamABCTracker', # name of the tracker
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

def tracker_init(weights_path, config):
    model = instantiate(config["model"])
        
    # replace_layers(model.connect_model.cls_dw, norm_lambda, contineous)
    # replace_layers(model.connect_model.reg_dw,  norm_lambda, contineous)
    
    # replace_layers(model.connect_model.bbox_tower,  norm_lambda, contineous)
    # replace_layers(model.connect_model.cls_tower,  norm_lambda, contineous)
    
    # print(model)
    # print(weights_path)
    
    model = load_from_lighting(model, weights_path, map_location=0).cuda().eval()
    SiamABC_tracker: SiamABCTracker = instantiate(config["tracker"], model=model)
    
    # setup tracker
    tracker = IdentityTracker(SiamABC_tracker)
    
    return tracker
    
def run_sequence(s, seq, num_gpu, experiment,  weights_path, config, dataset=None):


    tracker = tracker_init(weights_path, config)
    
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass
    
    seq_name = experiment.dataset.seq_names[s]
    print('--Sequence %d/%d: %s' % (s + 1, len(experiment.dataset), seq_name))
    # record_file = os.path.join(experiment.result_dir, tracker.name, '%s.txt' % seq_name)
    
    if dataset=='got10k':
        record_file = os.path.join(
                    experiment.result_dir, tracker.name, seq_name,
                    '%s_%03d.txt' % (seq_name, 0 + 1))
    else:
        record_file = os.path.join(experiment.result_dir, tracker.name, '%s.txt' % seq_name)
        
    if os.path.exists(record_file):
        print('  Found results, skipping', seq_name)
        return
    
    # print(seq)
    img_files, anno = seq[0]
    boxes, times = tracker.track(img_files, anno[0, :])
    # assert len(boxes) == len(anno)
    
    # record results
    experiment._record(record_file, boxes, times)
    
    sys.stdout.flush()


    exec_time = sum(times)
    num_frames = len(times)

    print('FPS: {}'.format(num_frames / exec_time))

    


def run(experiment, config, weights_path, num_gpus=1, threads=1, dataset=None):
    
       
    dataset_start_time = time.time()
    
    multiprocessing.set_start_method('spawn', force=True)

    param_list = [(s, seq, num_gpus, experiment, weights_path, config, dataset ) for s, seq in enumerate(product(experiment.dataset))]
    
    with multiprocessing.Pool(processes=threads) as pool:
        pool.starmap(run_sequence, param_list)
            
    print('Done, total time: {}'.format(str(timedelta(seconds=(time.time() - dataset_start_time)))))
    
    
        
        
@hydra.main(config_name="SiamABC_tracker", config_path="core/config")
def run_experiment(hydra_config: DictConfig):
    config = prepare_experiment(hydra_config)
    print("Experiment dir %s" % config["experiment"]["folder"])

    # weights_path = '/luna_data/zaveri/code/experiments/2024-02-22-16-00-25_Tracking_SiamABC_S_fast_mixed_att_at_neck/SiamABC/trained_model_ckpt_20.pt' #fast mixed att at neck
    # weights_path = '/luna_data/zaveri/code/experiments/2024-02-19-01-11-17_Tracking_SiamABC_S_mixed_fast_pol_att_w_corr_att_tn_trained_no_squeeze/SiamABC/trained_model_ckpt_17.pt' #fast polarized with corr att no squeeze
    # weights_path = '/luna_data/zaveri/code/experiments/2024-02-18-01-34-04_Tracking_SiamABC_S_mixed_polarized_att_w_corr_att_tn_trained/SiamABC/trained_model_ckpt_19.pt' #polarized att with corr att
    # weights_path = '/luna_data/zaveri/code/experiments/2024-02-17-16-25-31_Tracking_SiamABC_S_mixed_polarized_att_neck/SiamABC/trained_model_ckpt_15.pt' #polarized att
    # weights_path = '/luna_data/zaveri/code/experiments/2024-01-27-00-32-30_Tracking_SiamABC_cross_att_ssl_M/SiamABC/trained_model_ckpt_20.pt'
    # weights_path = '/luna_data/zaveri/code/experiments/2024-01-30-00-36-32_Tracking_SiamABC_cross_att_ssl_XL/SiamABC/trained_model_ckpt_22.pt'
    # weights_path = '/luna_data/zaveri/code/experiments/2023-10-29-21-02-32_Tracking_SiamABC_regmetX_full/SiamABC/trained_model_ckpt_15.pt'
    # weights_path = '/luna_data/zaveri/code/experiments/2024-03-07-00-55-11_Tracking_SiamABC/SiamABC/trained_model_ckpt_1.pt' # transitive loss
    # weights_path = '/luna_data/zaveri/code/experiments/2024-03-07-01-30-03_Tracking_SiamABC/SiamABC/trained_model_ckpt_1.pt' # regularizer loss
    
    # weights_path = '/luna_data/zaveri/code/experiments/2024-02-29-01-15-49_Tracking_SiamABC_M_fast_mixed_att_at_neck/SiamABC/trained_model_ckpt_22.pt' #20 works best
    
    # weights_path = '/luna_data/zaveri/code/experiments/2024-02-22-21-26-44_Tracking_SiamABC/SiamABC/trained_model_ckpt_20.pt'
    
    # # mobilenet vit
    # weights_path = '/luna_data/zaveri/code/experiments/2024-08-30-19-09-21_Tracking_SiamABC/SiamABC/trained_model_ckpt_20.pt'
    
    # self att
    weights_path = '/luna_data/zaveri/code/experiments/2024-08-30-17-51-48_Tracking_SiamABC/SiamABC/trained_model_ckpt_20.pt'
    
    config["model"]["model_size"] = 'S'
    config["tracker"]["dynamic_update"] = False
    config["tracker"]["N"] = 150
    config["tracker"]["smooth"] = False
    
    # got10k best 0.078 0.2 0.925
    config["tracker"]["penalty_k"] = 0.149 #0.096 #0.019 #0.026
    config["tracker"]["window_influence"] = 0.172 #0.166 #0.274 #0.241
    config["tracker"]["lr"] = 0.781 #0.565 #0.745 #0.766
    
    save_path = os.path.join(config["experiment"]["folder"], config["experiment"]["name"]+f'_lambda_{norm_lambda}_contineous_{contineous}')
    if os.path.exists(save_path) == False: os.makedirs(save_path)

    # experiment = ExperimentTrackingNet(
    #     root_dir=TRACKINGNET_DIR,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # run(experiment, config, weights_path, num_gpus=1, threads=8, dataset='trackingnet')
    # experiment.report(['SiamABCTracker'])
    
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])
    
    # # setup experiment (ITB)
    # experiment = ExperimentITB(
    #     root_dir=ITB_DIR,
    #     json_file=ITB_JSON,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # run(experiment, config, weights_path, num_gpus=1, threads=8)
    # experiment.report(['SiamABCTracker'])
    # # # experiment.run(tracker, visualize=False)
    # # # experiment.report([tracker.name])
    
    # setup experiment (AVIST)
    experiment = ExperimentAVIST(
        root_dir=AVIST_DIR,
        json_file=AVIST_JSON,
        result_dir=os.path.join(save_path,'results'),       # where to store tracking results
        report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    )
    run(experiment, config, weights_path, num_gpus=1, threads=8)
    experiment.report(['SiamABCTracker'])
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])

    # # setup experiment (TColor-128)
    # experiment = experiments.ExperimentTColor128(
    #     root_dir=TCOLOR128_DIR,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # run(experiment, config, weights_path, num_gpus=1, threads=8)
    # experiment.report(['SiamABCTracker'])
    # # experiment.run(tracker, visualize=False)
    # # experiment.report([tracker.name])


    # # setup experiment DTB170)
    # experiment = experiments.ExperimentDTB70(
    #     root_dir=DTB70_DIR,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # run(experiment, config, weights_path, num_gpus=1, threads=4)
    # experiment.report(['SiamABCTracker'])
    # # experiment.run(tracker, visualize=False)
    # # experiment.report([tracker.name])

# #     # NfS
#     experiment = experiments.ExperimentNfS(
#         root_dir=NFS_DIR,
#         result_dir=os.path.join(save_path,'results'),       # where to store tracking results
#         report_dir=os.path.join(save_path,'reports'),        # where to store evaluation reports
#         fps=30
#     )
#     run(experiment, config, weights_path, num_gpus=1, threads=8)
#     experiment.report(['SiamABCTracker'])
#     # experiment.run(tracker, visualize=False)
#     # experiment.report([tracker.name])

    # experiment = experiments.ExperimentGOT10k(
    #     root_dir=ROOT_DIR,          # GOT-10k's root directory
    #     subset='test',               # 'train' | 'val' | 'test'
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports'),        # where to store evaluation reports
    # )
    # run(experiment, config, weights_path, num_gpus=1, threads=8, dataset='got10k')
    # experiment.report(['SiamABCTracker'])

    # tracker = tracker_init(weights_path, config)
    # experiment.run(tracker, visualize=False, save_video=False)
    # experiment.report([tracker.name])

    # setup experiment (validation subset)
    # experiment = experiments.ExperimentGOT10k(
    #     root_dir=ROOT_DIR,          # GOT-10k's root directory
    #     subset='val',               # 'train' | 'val' | 'test'
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports'),        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False, save_video=False)
    # experiment.report([tracker.name])



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

    # lasot
    # # setup experiment (validation subset)
    experiment = experiments.ExperimentLaSOT(
        root_dir=LASOT_DIR,
        result_dir=os.path.join(save_path,'results'),       # where to store tracking results
        report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    )
    
    run(experiment, config, weights_path, num_gpus=1, threads=8)
    experiment.report(['SiamABCTracker'])
    # # # experiment.run(tracker, visualize=False)
    # # experiment.report([tracker.name])


    # # setup experiment (validation subset)
    # experiment = experiments.ExperimentVOT(
    #     root_dir=VOT_DIR,
    #     version=2018,
    #     experiments='supervised' ,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])

    # # setup experiment (validation subset)
    # experiment = experiments.ExperimentVOT(
    #     root_dir=VOT_DIR_2019,
    #     version=2019,
    #     experiments='supervised' ,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports')        # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])


    # # NfS
    # experiment = experiments.ExperimentNfS(
    #     root_dir=NFS_DIR,
    #     result_dir=os.path.join(save_path,'results'),       # where to store tracking results
    #     report_dir=os.path.join(save_path,'reports'),        # where to store evaluation reports
    #     fps=30
    # )
    # run(experiment, config, weights_path, num_gpus=1, threads=8)
    # experiment.report(['SiamABCTracker'])
    # # experiment.run(tracker, visualize=False)
    # # experiment.report([tracker.name])

if __name__ == '__main__':

    config = run_experiment()
    
    # for i in [0.1]: #, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
    #     norm_lambda=i
    #     contineous=False
    #     config = run_experiment()


    # for i in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
    #     norm_lambda=i
    #     contineous=True
    #     config = run_experiment()