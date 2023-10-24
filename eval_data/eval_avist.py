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
    temp = np.loadtxt(result_path, delimiter=',').astype(np.float)
    return np.array(temp)


def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T


def eval_avist_tune(result_path, json_path):
    list_path = json_path
    annos = json.load(open(list_path, 'r'))
    seqs = list(annos.keys())  # dict to list for py3
    n_seq = len(seqs)
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success_overlap = np.zeros((n_seq, 1, len(thresholds_overlap)))

    for i in range(n_seq):
        seq = seqs[i]
        gt_rect = np.array(annos[seq]['gt_rect']).astype(np.float)
        gt_center = convert_bb_to_center(gt_rect)
        bb = get_result_bb(result_path, seq)
        center = convert_bb_to_center(bb)
        success_overlap[i][0] = compute_success_overlap(gt_rect, bb)

    auc = success_overlap[:, 0, :].mean()
    return auc


if __name__ == "__main__":
    
    result_path = '/new_local_storage/zaveri/code/SiamABC/AVIST_penalty_k_0_4650_w_influence_0_1290_lr_0_0880'
    json_path = '/new_local_storage/zaveri/SOTA_Tracking_datasets/avist/AVIST.json'
    print(eval_avist_tune(result_path, json_path))
