import sys
import json
import os
import glob
from os.path import join, realpath, dirname
import numpy as np

# eval script for GOT10K validation dataset (official GOT10k only ranking according to tetsing dataset)
# use AUC not AO here


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def compute_success_overlap(gt_bb, result_bb):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    n_frame = len(gt_bb)
    success = np.zeros(len(thresholds_overlap))
    iou = overlap_ratio(gt_bb, result_bb)
    for i in range(len(thresholds_overlap)):
        success[i] = sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success


def compute_success_error(gt_center, result_center):
    thresholds_error = np.arange(0, 51, 1)
    n_frame = len(gt_center)
    success = np.zeros(len(thresholds_error))
    dist = np.sqrt(np.sum(np.power(gt_center - result_center, 2), axis=1))
    for i in range(len(thresholds_error)):
        success[i] = sum(dist <= thresholds_error[i]) / float(n_frame)
    return success



def get_result_bb(arch, seq):
    result_path = join(arch, seq + '.txt')
    temp = np.loadtxt(result_path, delimiter=',').astype("float")
    return np.array(temp)


def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T


def eval_auc(dataset='LASOTTEST', result_path='./test/', tracker_reg='S*', start=0, end=1e6):
    list_path = os.path.join(realpath(dirname(__file__)), '../../', 'dataset', dataset + '.json')
    annos = json.load(open(list_path, 'r'))
    seqs = list(annos.keys())  # dict to list for py3

    trackers = glob.glob(join(result_path, dataset, tracker_reg))
    trackers = trackers[start:min(end, len(trackers))]

    n_seq = len(seqs)
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    # thresholds_error = np.arange(0, 51, 1)

    success_overlap = np.zeros((n_seq, len(trackers), len(thresholds_overlap)))
    # success_error = np.zeros((n_seq, len(trackers), len(thresholds_error)))
    for i in range(n_seq):
        seq = seqs[i]
        gt_rect = np.array(annos[seq]['gt_rect']).astype("float")
        gt_center = convert_bb_to_center(gt_rect)
        for j in range(len(trackers)):
            tracker = trackers[j]
            print('{:d} processing:{} tracker: {}'.format(i, seq, tracker))
            bb = get_result_bb(tracker, seq)
            center = convert_bb_to_center(bb)
            success_overlap[i][j] = compute_success_overlap(gt_rect, bb)
            # success_error[i][j] = compute_success_error(gt_center, center)

    print('Success Overlap')

    max_auc = 0.
    max_name = ''
    for i in range(len(trackers)):
        auc = success_overlap[:, i, :].mean()
        if auc > max_auc:
            max_auc = auc
            max_name = trackers[i]
        print('%s(%.4f)' % (trackers[i], auc))

    print('\n%s Best: %s(%.4f)' % (dataset, max_name, max_auc))


def eval_lasot_tune(result_path, json_path):
    list_path = json_path
    annos = json.load(open(list_path, 'r'))
    seqs = list(annos.keys())  # dict to list for py3
    n_seq = len(seqs)
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success_overlap = np.zeros((n_seq, 1, len(thresholds_overlap)))
    thr_ce = np.arange(0, 51)
    prec_overlap = np.zeros((n_seq, 1, len(thr_ce)))
    
    for i in range(n_seq):
        seq = seqs[i]
        gt_rect = np.array(annos[seq]['gt_rect']).astype("float")
        gt_center = convert_bb_to_center(gt_rect)
        bb = get_result_bb(result_path, seq)
        center = convert_bb_to_center(bb)
        success_overlap[i][0] = compute_success_overlap(gt_rect, bb)
        prec_overlap[i][0] = compute_success_error(gt_center, center)

    auc = success_overlap[:, 0, :].mean()
    prec = prec_overlap[:, 0, 20].mean()
    succ_rate = success_overlap[:, 0, 10].mean()
    
    print(auc, prec, succ_rate)
    return auc
 

# if __name__ == "__main__":
    
#     # if len(sys.argv) < 5:
#     #     print('python ./lib/core/eval_lasot.py LASOTTEST ./result SiamFC* 0 1')
#     #     exit()
#     # dataset = sys.argv[1]
#     # result_path = sys.argv[2]
#     # tracker_reg = sys.argv[3]
#     # start = int(sys.argv[4])
#     # end = int(sys.argv[5])
#     # eval_auc(dataset, result_path, tracker_reg, start, end)
#     # # eval_auc('LASOTTEST', './result', 'DSiam', 0, 1)
    
#     # result_path = '/home/ramzav/ray_results/fitness_2024-02-26_03-30-05/fitness_ea464821_317_lambda=0.0000,lr=0.7850,penalty_k=0.0160,window_influence=0.2730_2024-02-28_22-26-55/SiamABC/LASOT_penalty_k_0_0160_w_influence_0_2730_lr_0_7850_AUC_0.6060509374555272' #S
    
#     result_path = '/home/ramzav/ray_results/fitness_2024-02-23_16-40-56/fitness_d650e4c0_148_lambda=0.0050,lr=0.7770,penalty_k=0.0550,window_influence=0.1700_2024-02-25_23-14-23/SiamABC/LASOT_penalty_k_0_0550_w_influence_0_1700_lr_0_7770_AUC_0.5717421948558175' #M
#     json_path = '/luna_data/zaveri/SOTA_Tracking_datasets/LaSOT/LASOTTEST.json'
#     print(eval_lasot_tune(result_path, json_path))

from got10k import experiments

if __name__ == "__main__":
    LASOT_DIR = '/new_local_storage/zaveri/SOTA_Tracking_datasets/LaSOT/LaSOT'
    result_path = '/home/ramzav/ray_results/fitness_2024-02-26_03-30-05/fitness_ea464821_317_lambda=0.0000,lr=0.7850,penalty_k=0.0160,window_influence=0.2730_2024-02-28_22-26-55/SiamABC/LASOT_penalty_k_0_0160_w_influence_0_2730_lr_0_7850_AUC_0.6060509374555272' 
    json_path = '/new_local_storage/zaveri/SOTA_Tracking_datasets/LaSOT/LASOTTEST.json'
    eval_lasot_tune(result_path, json_path)
    
    # # eval_auc('LASOTTEST', './result', 'DSiam', 0, 1)
    
    # # result_path = '/home/ramzav/ray_results/fitness_2024-02-26_03-30-05/fitness_ea464821_317_lambda=0.0000,lr=0.7850,penalty_k=0.0160,window_influence=0.2730_2024-02-28_22-26-55/SiamABC/ ' #S
    
    # result_path = '/home/ramzav/ray_results/fitness_2024-02-23_16-40-56/fitness_d650e4c0_148_lambda=0.0050,lr=0.7770,penalty_k=0.0550,window_influence=0.1700_2024-02-25_23-14-23/SiamABC/LASOT_penalty_k_0_0160_w_influence_0_2730_lr_0_7850_AUC_0.6060509374555272' 
    
    # experiment = experiments.ExperimentLaSOT(LASOT_DIR)
    # experiment.result_dir=result_path
    
    # # experiment.report(['LASOT_penalty_k_0_0160_w_influence_0_2730_lr_0_7850_AUC_0.6060509374555272'])
    # experiment.report(['LASOT_penalty_k_0_0550_w_influence_0_1700_lr_0_7770_AUC_0.5717421948558175'])