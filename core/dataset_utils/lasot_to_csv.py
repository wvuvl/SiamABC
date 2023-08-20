import os
import pandas as pd
import numpy as np
import cv2
import csv
from tqdm import tqdm
from collections import OrderedDict

def printBB(dir, frames_folder, BB_file):

    ArrayBB = np.loadtxt(BB_file, delimiter=",")  

    frames_list=[os.path.join(dir, frame) for frame in os.listdir(frames_folder) if frame.endswith(".jpg") ]

    if ( not len(ArrayBB) == len(frames_list)):
        #print("Not the same number of frames and annotation!" ) 
        if (np.ndim(ArrayBB) == 1):
            tmp = ArrayBB
            del ArrayBB
            ArrayBB = [[]]
            ArrayBB[0] = tmp


    return ArrayBB

    
root = '/data/zaveri/SOTA_Tracking_datasets/LaSOT/LaSOT'


file_path = os.path.join(root, 'training_set.txt')
sequence_list = pd.read_csv(file_path, header=None, squeeze=True).values.tolist()




data = []

for idx, seq_name in tqdm(enumerate(sequence_list)):
    
    class_name = seq_name.split('-')[0]
    vid_id = seq_name.split('-')[1]
        
    video_path = os.path.join(root, class_name, class_name + '-' + vid_id)
    frames_list=[os.path.join(video_path, 'img',frame) for frame in os.listdir(os.path.join(video_path, 'img')) if frame.endswith(".jpg") ]
    
    video_gt_path = os.path.join(video_path, "groundtruth.txt")
    occlusion_file = os.path.join(video_path, "full_occlusion.txt")
    out_of_view_file = os.path.join(video_path, "out_of_view.txt")
    
    video_gt = printBB(root, video_path, video_gt_path)
    
    with open(occlusion_file, 'r', newline='') as f:
        occlusion = [int(v) for v in list(csv.reader(f))[0]]
    with open(out_of_view_file, 'r') as f:
        out_of_view = [int(v) for v in list(csv.reader(f))[0]]
        
        
    for frame_num, (frame, anno, occ, oov) in enumerate(zip(sorted(frames_list),video_gt, occlusion, out_of_view)):
        img_path = frame
        img = cv2.imread(img_path)
        
        x = int(anno[0])
        y = int(anno[1])
        w = int(anno[2])
        h = int(anno[3])
        
        bbox_exist = 1 if (occ==0 and oov==0)  else 0
        bbox_border = int(x<=0 or y<=0 or (x+w)>=img.shape[1]-1 or (y+h)>=img.shape[0]-1)
        
        x = x if x>0 else 0
        y = y if y>0 else 0
        w = w if (x+w)< img.shape[1] else img.shape[1]-1
        h = h if (y+h)< img.shape[0] else img.shape[0]-1
        
        data.append([str(idx), class_name, frame_num, img_path, [x,y,w,h], [img.shape[1], img.shape[0]], 'lasot',bbox_exist, bbox_border])
    # sanity save
    df = pd.DataFrame(data, columns=["sequence_id","track_id","frame_index","img_path","bbox","frame_shape","dataset","presence","near_corner"])
    df.to_csv("lasot.csv")

df = pd.DataFrame(data, columns=["sequence_id","track_id","frame_index","img_path","bbox","frame_shape","dataset","presence","near_corner"])
df.to_csv("lasot.csv")
        
print(data)