from __future__ import absolute_import, division, print_function

import os
import numpy as np
import glob
import ast
import json
import time
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import cv2

from got10k.datasets import TrackingNet
from got10k.utils.metrics import rect_iou
from got10k.utils.viz import show_frame
from got10k.utils.ioutils import compress


    
class ExperimentTrackingNet(object):

    def __init__(self, root_dir, subset='test', list_file=None,
                 result_dir='results', report_dir='reports', use_dataset=True):
        super(ExperimentTrackingNet, self).__init__()
        assert subset in ['test']
        self.subset = subset
        if use_dataset:
            self.dataset = TrackingNet(
                root_dir, subset=subset)
        self.result_dir = os.path.join(result_dir, 'TrackingNet')
        self.report_dir = os.path.join(report_dir, 'TrackingNet')

    def run(self, tracker, visualize=False, save_video=False, overwrite_result=True):
        if self.subset == 'test':
            print('\033[93m[WARNING]:\n' \
                  'The groundtruths of Trackingnet\'s test set is withholded.\n' \
                )
            time.sleep(2)

        print('Running tracker %s on TrackingNet...' % tracker.name)
        self.dataset.return_meta = False

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (
                s + 1, len(self.dataset), seq_name))

            # run multiple repetitions for each sequence
            for r in range(1):
                # check if the tracker is deterministic
                if r > 0 and tracker.is_deterministic:
                    break
                elif r == 3 and self._check_deterministic(
                    tracker.name, seq_name):
                    print('  Detected a deterministic tracker, ' +
                          'skipping remaining trials.')
                    break
                print(' Repetition: %d' % (r + 1))

                # skip if results exist
                record_file = os.path.join( self.result_dir, tracker.name, '%s.txt' % (seq_name))
                if os.path.exists(record_file) and not overwrite_result:
                    print('  Found results, skipping', seq_name)
                    continue

                # tracking loop
                boxes, times = tracker.track(
                    img_files, anno[0, :], visualize=visualize)
                
                # record results
                self._record(record_file, boxes, times)

            # save videos
            if save_video:
                video_dir = os.path.join(os.path.dirname(os.path.dirname(self.result_dir)),
                    'videos', 'TrackingNet', tracker.name)
                video_file = os.path.join(video_dir, '%s.avi' % seq_name)

                if not os.path.isdir(video_dir):
                    os.makedirs(video_dir)
                image = Image.open(img_files[0])
                img_W, img_H = image.size
                out_video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MJPG'), 10, (img_W, img_H))
                for ith, (img_file, pred) in enumerate(zip(img_files, boxes)):
                    image = Image.open(img_file)
                    if not image.mode == 'RGB':
                        image = image.convert('RGB')
                    img = np.array(image)[:, :, ::-1].copy()
                    pred = pred.astype(int)
                    cv2.rectangle(img, (pred[0], pred[1]), (pred[0] + pred[2], pred[1] + pred[3]), (0, 255, 0), 2)
                    if ith < anno.shape[0]:
                        gt = anno[ith].astype(int)
                        cv2.rectangle(img, (gt[0], gt[1]), (gt[0] + gt[2], gt[1] + gt[3]), (0, 0, 255), 2)
                    out_video.write(img)
                out_video.release()
                print('  Videos saved at', video_file)

    def report(self, tracker_names):
        assert isinstance(tracker_names, (list, tuple))

        if self.subset == 'test':
            pwd = os.getcwd()

            # generate compressed submission file for each tracker
            for tracker_name in tracker_names:
                # compress all tracking results
                result_dir = os.path.join(self.result_dir, tracker_name)
                os.chdir(result_dir)
                save_file = '../%s' % tracker_name
                compress('.', save_file)
                print('Records saved at', save_file + '.zip')

            # switch back to previous working directory
            os.chdir(pwd)

            return None

   
    def _record(self, record_file, boxes, times):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        while not os.path.exists(record_file):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        print('  Results recorded at', record_file)

        # record running times
        # time_file = record_file[:record_file.rfind('_')] + '_time.txt'
        # times = times[:, np.newaxis]
        # if os.path.exists(time_file):
        #     exist_times = np.loadtxt(time_file, delimiter=',')
        #     if exist_times.ndim == 1:
        #         exist_times = exist_times[:, np.newaxis]
        #     times = np.concatenate((exist_times, times), axis=1)
        # np.savetxt(time_file, times, fmt='%.8f', delimiter=',')

    def _check_deterministic(self, tracker_name, seq_name):
        record_dir = os.path.join(
            self.result_dir, tracker_name, seq_name)
        record_files = sorted(glob.glob(os.path.join(
            record_dir, '%s_[0-9]*.txt' % seq_name)))

        if len(record_files) < 3:
            return False

        records = []
        for record_file in record_files:
            with open(record_file, 'r') as f:
                records.append(f.read())
        
        return len(set(records)) == 1


    