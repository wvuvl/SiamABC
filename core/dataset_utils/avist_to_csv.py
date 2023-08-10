import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np


def load_annotation_file_otb(ann_file):
    bboxes = []
    frame_num = 0
    with open(ann_file) as f:
        data = f.read().rstrip().split('\n')
        for bb in data:
            if ',' in bb: x1, y1, w, h = [int(i) for i in bb.split(',')]
            else: x1, y1, w, h = [int(float(i)) for i in bb.split()]
            #x1, y1, w, h
            bboxes.append([x1, y1, w, h, frame_num])
            frame_num+=1

    return bboxes


video_dir = '/data/zaveri/dataset/avist/sequences'
gt_dir = '/data/zaveri/dataset/avist/anno/'
out_of_view = '/data/zaveri/dataset/avist/out_of_view'
full_occlusions = '/data/zaveri/dataset/avist/full_occlusion'

data = []

for idx, video in tqdm(enumerate(os.listdir(video_dir))):
    
    video_name = os.path.splitext(video)[0]
    video_path = os.path.join(video_dir, video)
    
    gt_path = os.path.join(gt_dir,video_name+'.txt')
    gt = load_annotation_file_otb(gt_path)
    
    video_out_of_view = os.path.join(out_of_view,video_name+'_out_of_view.txt')  
    with open(video_out_of_view) as f:
        lines = f.readlines()
        video_out_of_view_frames = lines[0].split(',')
                
    video_full_occlusions = os.path.join(full_occlusions, video_name+'_full_occlusion.txt') 
    with open(video_full_occlusions) as f:
        lines = f.readlines()
        video_full_occlusions_frames = lines[0].split(',')


    frames = sorted(os.listdir(video_path))
    for anno_idx, anno in enumerate(gt):
        img_path = os.path.join(video_path, frames[anno_idx])
        img = cv2.imread(img_path)
    
        bbox_border = 0
        if 0 in anno[:4]: bbox_border = 1
        if img.shape[1]-1 in anno[:4]: bbox_border = 1
        if img.shape[0]-1 in anno[:4]: bbox_border = 1

        bbox_exist = 1
        if int(video_full_occlusions_frames[anno_idx]) == 1 or int(video_out_of_view_frames[anno_idx]) == 1: bbox_exist = 0
        
        data.append([str(idx), video, anno[4], img_path, anno[:4], [img.shape[1], img.shape[0]], 'got10k',bbox_exist, bbox_border])


df = pd.DataFrame(data, columns=["sequence_id","track_id","frame_index","img_path","bbox","frame_shape","dataset","presence","near_corner"])
df.to_csv("AVIST.csv")
        
print(data)