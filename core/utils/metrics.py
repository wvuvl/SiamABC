""" source: https://github.com/got-10k/toolkit """

import numpy as np
from got10k.experiments import ExperimentOTB
def rect_iou(rects1, rects2, bound=None):
    r"""Intersection over union.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert rects1.shape == rects2.shape

        
    if bound is not None:
        # bounded rects1
        rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # bounded rects2
        rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    rects_inter = intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious

def center_error(rects1, rects2):
    r"""Center error.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    centers1 = rects1[..., :2] + (rects1[..., 2:] - 1) / 2
    centers2 = rects2[..., :2] + (rects2[..., 2:] - 1) / 2
    errors = np.sqrt(np.sum(np.power(centers1 - centers2, 2), axis=-1))

    return errors

def intersection(rects1, rects2):
    r"""Rectangle intersection.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                    rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                    rects2[..., 1] + rects2[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T

def calc_curves(ious, center_errors, nbins_iou=21, nbins_ce=51):
    ious = np.asarray(ious, float)[:, np.newaxis]
    center_errors = np.asarray(center_errors, float)[:, np.newaxis]

    thr_iou = np.linspace(0, 1, nbins_iou)[np.newaxis, :]
    thr_ce = np.arange(0, nbins_ce)[np.newaxis, :]

    bin_iou = np.greater(ious, thr_iou)
    bin_ce = np.less_equal(center_errors, thr_ce)

    succ_curve = np.mean(bin_iou, axis=0)
    prec_curve = np.mean(bin_ce, axis=0)

    return succ_curve, prec_curve

def calc_metrics(boxes, anno):
    # can be modified by children classes
    ious = rect_iou(boxes, anno)
    center_errors = center_error(boxes, anno)
    return ious, center_errors

if __name__ == '__main__':
    x = np.array([[1,2,3,4], [1,2,3,4]])
    y = np.array([[1,2,3,4], [2,3,4,5]])

            
    ious, center_errors = calc_metrics(x,y)
    print(ious, center_errors)
    print(calc_curves(ious, center_errors))