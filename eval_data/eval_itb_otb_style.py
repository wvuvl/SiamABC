from __future__ import absolute_import, division, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from PIL import Image
import json
import os
import numpy as np

from got10k.experiments.otb import ExperimentOTB
from got10k.utils.metrics import rect_iou, center_error
from got10k.utils.viz import show_frame

class ITB(object):
    def __init__(self, root_dir, json_file):
        super(ITB, self).__init__()
        self.annos = json.load(open(json_file, 'r'))
        self.seq_names = list(self.annos.keys())  # dict to list for py3
        self.root_dir = root_dir
    
    def __getitem__(self, index):
        img_files = [os.path.join(self.root_dir,  self.annos[self.seq_names[index]]['video_dir'], im_f) for im_f in self.annos[self.seq_names[index]]['img_names']]
        anno = np.array(self.annos[self.seq_names[index]]['gt_rect']) # - [1, 1, 0, 0]
        return img_files, anno

    def __len__(self):
        return len(self.seq_names)

        
class ExperimentITB(ExperimentOTB):

    def __init__(self, root_dir, json_file, version='ITB',
                 result_dir='results', report_dir='reports'):
        
        self.dataset = ITB(root_dir, json_file)
        self.result_dir = os.path.join(result_dir, version.upper())
        self.report_dir = os.path.join(report_dir, version.upper())
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51

    def _calc_metrics(self, boxes, anno):
        valid = ~np.any(np.isnan(anno), axis=1)
        if len(valid) == 0:
            print('Warning: no valid annotations')
            return None, None
        else:
            ious = rect_iou(boxes[valid, :], anno[valid, :])
            center_errors = center_error(
                boxes[valid, :], anno[valid, :])
            return ious, center_errors