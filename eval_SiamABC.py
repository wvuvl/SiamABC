import os
import random

from core.utils.utils import read_img
from eval_data.utils import  load_dataset, poly_iou, get_axis_aligned_bbox, center2xywh

from eval_toolkit.pysot.datasets import VOTDataset
from eval_toolkit.pysot.evaluation import EAOBenchmark
from eval_data import eval_otb, eval_got10k, eval_lasot, eval_nfs, eval_uav123, eval_avist
from tqdm import tqdm

# -----------------------------------------------
def track_tune(tracker, video, dataset_name, penalty_k, window_influence, lr):
    
    benchmark_name = dataset_name
    tracker_path = os.path.join('SiamABC', (benchmark_name +
                                     f'_penalty_k_{penalty_k:.4f}' +
                                     f'_w_influence_{window_influence:.4f}' +
                                     f'_lr_{lr:.4f}').replace('.', '_'))  # no .
    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in benchmark_name:
        baseline_path = os.path.join(tracker_path, 'baseline')
        video_path = os.path.join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    elif 'GOT10K' in benchmark_name:
        re_video_path = os.path.join(tracker_path, video['name'])
        if not os.path.exists(re_video_path): os.makedirs(re_video_path)
        result_path = os.path.join(re_video_path, '{:s}.txt'.format(video['name']))
    elif 'NFS' in benchmark_name or 'UAV' in benchmark_name:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['video_dir']))
    else:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

    # occ for parallel running
    if not os.path.exists(result_path):
        fin = open(result_path, 'w')
        fin.close()
    else:
        if benchmark_name.startswith('OTB'):
            print('results exist')
            return tracker_path
        elif benchmark_name.startswith('VOT') or benchmark_name.startswith('GOT10K') or benchmark_name.startswith('LASOT') or benchmark_name.startswith('NFS') or benchmark_name.startswith('UAV') or benchmark_name.startswith('AVIST') :
            print('results exist')
            return 0
        else:
            print('benchmark not supported now')
            return

    start_frame, lost_times, toc = 0, 0, 0

    regions = []  # result and states[1 init / 2 lost / 0 skip]

    # for rgbt splited test

    image_files, gt = video['image_files'], video['gt']
    
    for f, image_file in enumerate(image_files):
        im = read_img(image_file)
        # print(gt[f])
        if f == start_frame:  # init
            tracker.initialize(im, center2xywh(get_axis_aligned_bbox(gt[f])))  # init tracker
            regions.append([float(1)] if 'VOT' in benchmark_name else gt[f])
        
        elif f > start_frame:  # tracking
            
            location, _ = tracker.update(im)  # track
            b_overlap = poly_iou(gt[f], location) if 'VOT' in benchmark_name else 1
            
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append([float(2)])
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        
        else:  # skip
            regions.append([float(0)])

    # save results for OTB
    if 'OTB' in benchmark_name or 'GOT10K' in benchmark_name or 'LASOT' in benchmark_name or 'NFS' in benchmark_name or 'UAV' in benchmark_name or 'AVIST' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')

    elif 'VOT' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')

    if 'OTB' in benchmark_name or 'VOT' in benchmark_name or 'GOT10K' in benchmark_name or 'LASOT' in benchmark_name or 'NFS' in benchmark_name or 'UAV' in benchmark_name  or 'AVIST' in benchmark_name:
        return tracker_path
    else:
        print('benchmark not supported now 2')


def auc_avist(tracker, dataset_name, data_path, penalty_k, window_influence, lr, base_path, json_path=None):
    """
    get AUC for AVIST benchmark
    """
    dataset = load_dataset(dataset_name, base_path=base_path, json_path=json_path)
    video_keys = list(dataset.keys()).copy()
    random.shuffle(video_keys)

    for video in video_keys: #tqdm(video_keys, ncols=100):
        result_path = track_tune(tracker, dataset[video], dataset_name, penalty_k, window_influence, lr)

    auc = eval_avist.eval_avist_tune(result_path, json_path)

    os.rename(result_path, result_path+'_AUC_'+str(auc))
    return auc

def auc_uav123(tracker, dataset_name, data_path, penalty_k, window_influence, lr, base_path, json_path=None):
    """
    get AUC for UAV123 benchmark
    """
    dataset = load_dataset(dataset_name, base_path=base_path, json_path=json_path)
    video_keys = list(dataset.keys()).copy()
    random.shuffle(video_keys)

    for video in video_keys: #tqdm(video_keys, ncols=100):
        result_path = track_tune(tracker, dataset[video], dataset_name, penalty_k, window_influence, lr)

    auc = eval_uav123.eval_uav123_tune(result_path, json_path)

    os.rename(result_path, result_path+'_AUC_'+str(auc))
    return auc

def auc_nfs(tracker, dataset_name, data_path, penalty_k, window_influence, lr, base_path, json_path=None):
    """
    get AUC for NFS benchmark
    """
    dataset = load_dataset(dataset_name, base_path=base_path, json_path=json_path)
    video_keys = list(dataset.keys()).copy()
    random.shuffle(video_keys)

    for video in video_keys: #tqdm(video_keys, ncols=100):
        result_path = track_tune(tracker, dataset[video], dataset_name, penalty_k, window_influence, lr)

    auc = eval_nfs.eval_nfs_tune(result_path, json_path)

    os.rename(result_path, result_path+'_AUC_'+str(auc))
    return auc


def auc_lasot(tracker, dataset_name, data_path, penalty_k, window_influence, lr, base_path, json_path=None):
    """
    get AUC for LaSoT benchmark
    """
    dataset = load_dataset(dataset_name, base_path=base_path, json_path=json_path)
    video_keys = list(dataset.keys()).copy()
    random.shuffle(video_keys)

    for video in video_keys: #tqdm(video_keys, ncols=100):
        result_path = track_tune(tracker, dataset[video], dataset_name, penalty_k, window_influence, lr)

    auc = eval_lasot.eval_lasot_tune(result_path, json_path)

    os.rename(result_path, result_path+'_AUC_'+str(auc))
    return auc



def auc_got10k(tracker, dataset_name, data_path, penalty_k, window_influence, lr, base_path, json_path=None):
    """
    get AUC for got10k benchmark
    """
    dataset = load_dataset(dataset_name, base_path=base_path, json_path=json_path)
    video_keys = list(dataset.keys()).copy()
    random.shuffle(video_keys)

    for video in video_keys: #tqdm(video_keys, ncols=100):
        result_path = track_tune(tracker, dataset[video], dataset_name, penalty_k, window_influence, lr)

    auc = eval_got10k.eval_got10k_tune(result_path, json_path)

    os.rename(result_path, result_path+'_AUC_'+str(auc))
    return auc


def auc_otb(tracker, dataset_name, data_path, penalty_k, window_influence, lr, base_path, json_path=None):
    """
    get AUC for OTB benchmark
    """
    dataset = load_dataset(dataset_name, base_path=base_path, json_path=json_path)
    video_keys = list(dataset.keys()).copy()
    random.shuffle(video_keys)

    for video in video_keys: #tqdm(video_keys, ncols=100):
        result_path = track_tune(tracker, dataset[video], dataset_name, penalty_k, window_influence, lr)

    auc = eval_otb.eval_auc_tune(result_path, json_path)

    os.rename(result_path, result_path+'_AUC_'+str(auc), )
    return auc

def eao_vot(tracker, dataset_name, data_path, penalty_k, window_influence, lr, base_path, json_path=None):
    
    dataset = load_dataset(dataset_name, base_path=base_path, json_path=json_path)
    video_keys = sorted(list(dataset.keys()).copy())

    
    for video in video_keys: # tqdm(video_keys, ncols=100):
        result_path = track_tune(tracker, dataset[video], dataset_name, penalty_k, window_influence, lr)

    re_path = os.path.dirname(result_path) #result_path.split('/')[0]
    tracker = result_path.split('/')[-1]

    dataset = VOTDataset(dataset_name, data_path)

    dataset.set_tracker(re_path, tracker)
    benchmark = EAOBenchmark(dataset)
    eao = benchmark.eval(tracker)
    eao = eao[tracker]['all']

    os.rename(result_path, result_path+'_EAO_'+str(eao))
    
    return eao


# result_path = '/new_local_storage/zaveri/code/SiamABC/SiamABC/VOT2019_penalty_k_0_0981_w_influence_0_5873_lr_0_4443'

# re_path = '/new_local_storage/zaveri/code/SiamABC/SiamABC'
# tracker = result_path.split('/')[-1]
    
# dataset = VOTDataset('VOT2019', '/new_local_storage/zaveri/SOTA_Tracking_datasets/vot2019')

# print(re_path)
# dataset.set_tracker(re_path, tracker)
# benchmark = EAOBenchmark(dataset)
# print(dataset)
# print(tracker)
# eao = benchmark.eval(tracker)
# eao = eao[tracker]['all']

# print(eao)