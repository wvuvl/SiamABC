import os
import pandas as pd
import cv2
from tqdm import tqdm

def load_annotation_file_otb(ann_file):
    bboxes = []
    frame_num = 0
    with open(ann_file) as f:
        data = f.read().rstrip().split('\n')
        for bb in data:
            if ',' in bb: x1, y1, w, h = [int(i) for i in bb.split(',')]
            else: x1, y1, w, h = [int(float(i)) for i in bb.split()]
            bboxes.append([x1, y1, x1+w, y1+h, frame_num])
            frame_num+=1

    return bboxes

path = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/got-10k/OTB'

data = []

for idx, video in tqdm(enumerate(os.listdir(path))):
    video_path = os.path.join(path, video,'img')
    video_gt_path = os.path.join(path,video,'groundtruth_rect.txt')
    video_gt = load_annotation_file_otb(video_gt_path)
    
    
    for frame, anno in zip(sorted(os.listdir(video_path)),video_gt):
        img_path = os.path.join(video_path, frame)
        img = cv2.imread(img_path)
        
        bbox_border = 0
        if 0 in anno[:4]: bbox_border = 1
        if img.shape[1]-1 in anno[:4]: bbox_border = 1
        if img.shape[0]-1 in anno[:4]: bbox_border = 1
        
        bbox_exist = 1
        if anno[0] == 0 and anno[1] == 0 and anno[2] == 0 and anno[3] == 0: bbox_exist=0
        
        data.append([str(idx), video, anno[4], img_path, anno[:4], [img.shape[1], img.shape[0]], 'got10k',bbox_exist, bbox_border])

df = pd.DataFrame(data, columns=["sequence_id","track_id","frame_index","img_path","bbox","frame_shape","dataset","presence","near_corner"])
df.to_csv("got-10k.csv")
        
print(data)