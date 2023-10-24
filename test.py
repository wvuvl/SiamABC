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
import matplotlib
import matplotlib.pyplot as plt
from core.utils.utils import read_img, get_iou, plot_loss
from core.train import get_tracking_test_datasets
from core.train.train_val import get_collate_for_dataset
from core.utils import prepare_experiment, create_logger
from core.utils.torch_stuff import load_from_lighting
import core.utils.eval_otb as eval_otb
import core.utils.metrics  as metrics

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
        draw = draw_bbox(frame, bbox)
        visualized_frames.append(cv2.resize(draw, (1280, 720)))
    return visualized_frames


def test_network(tracker, data_loader, save_path, threshold:int = 0.5):

    
    vid_len = data_loader.__len__()
    eao_all, robustness_all = [], []
    df = []
    
    
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success_overlap = np.zeros((vid_len, len(thresholds_overlap)))
    
    thresholds_error = np.arange(0, 51, 1)
    success_error = np.zeros((vid_len, len(thresholds_error)))
    
    nbins_iou=21
    nbins_ce=51
    succ_curve = np.zeros((vid_len, nbins_iou)) # for # of bins, default: 21
    prec_curve = np.zeros((vid_len, nbins_ce)) # for # of bins, default: 51
    speeds = np.zeros(vid_len)
    
    
    seq_ious = []
    progress_bar = tqdm(data_loader)
    for index, batch in enumerate(progress_bar):
        for image_files, annotations, dataset_name in batch:
            ious = []            
            image_t_0 = read_img(image_files[0])
            tracker.initialize(image_t_0, list(map(int, annotations[0])))
            num_samples =min(200, len(annotations)) if dataset_name=='lasot' or dataset_name=='nfs'  or dataset_name=='trackingnet' else len(annotations)
            annotations = annotations[:num_samples]
            video = [image_t_0]
            tracked_bboxes = [annotations[0].astype('int')]
            for i in range(1, num_samples):
                search_image = read_img(image_files[i])
                video.append(search_image)
                bbox, cls_score = tracker.update(search=search_image)
                tracked_bboxes.append(np.array(bbox).astype('int'))
                if len(annotations) > 1:
                    iou = get_iou(np.array(bbox), np.array(list(map(int, annotations[i]))))
                    ious.append(iou)
                
                    
            # visualized_video = visualize(video, tracked_bboxes)
            # if os.path.exists(os.path.join(save_path,dataset_name))==False: os.makedirs(os.path.join(save_path,dataset_name)) 
            # iio.imwrite(
            #     os.path.join(save_path,dataset_name, str(index)+'_'+image_files[0].split(os.sep)[-2]+'.mp4'), 
            #     visualized_video, 
            #     fps=240 if dataset_name=='nfs' else 30)

            if len(ious) >0:
                mean_iou = np.mean(ious)
                seq_ious.append(mean_iou)
                progress_bar.set_description(f'Mean IoU - {np.mean(seq_ious)}', )
                
                # anno = np.array(annotations)
                # boxes = np.array(tracked_bboxes)
                # success_overlap[index] = eval_otb.compute_success_overlap(anno,boxes )
                    
                # gt_center = eval_otb.convert_bb_to_center(anno)
                # bb_center = eval_otb.convert_bb_to_center(boxes)
                # success_error[index] = eval_otb.compute_success_error(gt_center, bb_center)
                # logger.info('success_overlap: {:0.3f}, success_error: {:0.3f}'.format(success_overlap[index].mean(), success_error[index][20]))
                
                # # otb org
                # ious, center_errors = metrics.calc_metrics(boxes, anno)
                # succ_curve[index], prec_curve[index] = metrics.calc_curves(ious, center_errors, nbins_iou=nbins_iou, nbins_ce=nbins_ce)
                # logger.info(f'success_score: {np.mean(succ_curve[index])}, precision_score: {prec_curve[index][20]}')
        
    if len(seq_ious) > 0:
        logger.info(f'Mean IoU - {np.mean(seq_ious)}' )
        logger.info('AUC: {:0.3f}, precision: {:0.3f}'.format(success_overlap.mean(), np.mean(success_error, axis=0)[20]))
        
        # # otb org
        # succ_curve = np.mean(succ_curve, axis=0)
        # prec_curve = np.mean(prec_curve, axis=0)
        # succ_score = np.mean(succ_curve)
        # prec_score = prec_curve[20]
        # succ_rate = succ_curve[nbins_iou // 2]
        # logger.info(f'succ_score: {succ_score}, prec_score: {prec_score}, succ_rate: {succ_rate}')
        # plot_curves(nbins_iou, nbins_ce, succ_curve, prec_curve, succ_score, prec_score, save_path)

def plot_curves(nbins_iou, nbins_ce, succ_curve, prec_curve, succ_score, prec_score, save_path):
    

    succ_file = os.path.join(save_path, 'success_plots.png')
    prec_file = os.path.join(save_path, 'precision_plots.png')
    
    # markers
    markers = ['-', '--', '-.']
    markers = [c + m for m in markers for c in [''] * 10]

    # plot success curves
    thr_iou = np.linspace(0, 1, nbins_iou)
    fig, ax = plt.subplots()
    lines = []
    legends = []
    for i, name in enumerate(['GOTURN']):
        line, = ax.plot(thr_iou,
                        succ_curve,
                        markers[i % len(markers)])
        lines.append(line)
        legends.append('%s: [%.3f]' % (name, succ_score))
    matplotlib.rcParams.update({'font.size': 7.4})
    legend = ax.legend(lines, legends, loc='center left',
                        bbox_to_anchor=(1, 0.5))

    matplotlib.rcParams.update({'font.size': 9})
    ax.set(xlabel='Overlap threshold',
            ylabel='Success rate',
            xlim=(0, 1), ylim=(0, 1),
            title='Success plots of OPE')
    ax.grid(True)
    fig.tight_layout()
    
    logger.info('Saving success plots to '+succ_file)
    fig.savefig(succ_file,
                bbox_extra_artists=(legend,),
                bbox_inches='tight',
                dpi=300)

    # plot precision curves
    thr_ce = np.arange(0, nbins_ce)
    fig, ax = plt.subplots()
    lines = []
    legends = []
    for i, name in enumerate(['GOTURN']):
        line, = ax.plot(thr_ce,
                        prec_curve,
                        markers[i % len(markers)])
        lines.append(line)
        legends.append('%s: [%.3f]' % (name, prec_score))
    matplotlib.rcParams.update({'font.size': 7.4})
    legend = ax.legend(lines, legends, loc='center left',
                        bbox_to_anchor=(1, 0.5))
    
    matplotlib.rcParams.update({'font.size': 9})
    ax.set(xlabel='Location error threshold',
            ylabel='Precision',
            xlim=(0, thr_ce.max()), ylim=(0, 1),
            title='Precision plots of OPE')
    ax.grid(True)
    fig.tight_layout()
    
    logger.info('Saving precision plots to '+ prec_file)
    fig.savefig(prec_file, dpi=300)
        
def test(config: Dict[str, Any], save_path, weights_path) -> None:
        
    model = instantiate(config["model"])
    model = load_from_lighting(model, weights_path, map_location=0).cuda().eval()
    
    tracker: AEVTTracker = instantiate(config["tracker"], model=model)

    test_dataset = get_tracking_test_datasets(config)
    assert test_dataset is not None, "Test Dataset - None"

    collate_fn = get_collate_for_dataset(test_dataset)
    test_dl = DataLoader(
                dataset=test_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn
            )
    
    test_network(tracker, test_dl, save_path)


@hydra.main(config_name="AEVT_tracker", config_path="core/config")
def run_experiment(hydra_config: DictConfig) -> None:
    config = prepare_experiment(hydra_config)
    logger.info("Experiment dir %s" % config["experiment"]["folder"])
    
    save_path = os.path.join(config["experiment"]["folder"], config["experiment"]["name"])
    if os.path.exists(save_path) == False: os.makedirs(save_path)

    weights_path = '/new_local_storage/zaveri/code/experiments/2023-10-17-19-38-53_Tracking_AEVT/AEVT/trained_model_ckpt_2.pt'
    test(config, save_path, weights_path)
            

if __name__ == "__main__":
    run_experiment()