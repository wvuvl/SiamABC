import os
import pandas as pd
import numpy as np
import cv2
import csv
from PIL import Image
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

    
root = '/new_local_storage/zaveri/SOTA_Tracking_datasets/GOT10k/GOT10k/train'


file_path = os.path.join(root, 'list.txt')
with open(os.path.join(file_path)) as f:
    sequence_list = list(csv.reader(f))
sequence_list = [dir_name[0] for dir_name in sequence_list]


all_categories = '/new_local_storage/zaveri/code/SiamABC/core/dataset_utils/data_specs/got10k_train_full_split.txt'
vot_exclude = '/new_local_storage/zaveri/code/SiamABC/core/dataset_utils/data_specs/got10k_vot_exclude.txt'
# valid_vot_categories = '/data/zaveri/code/SiamABC/core/dataset_utils/data_specs/got10k_vot_train_split.txt'

seq_ids =np.array(pd.read_csv(all_categories, header=None).values.tolist(), dtype=np.int64).squeeze()
sequence_list = [sequence_list[i] for i in seq_ids]

vot_excluded_seqs =np.array(pd.read_csv(vot_exclude, header=None).values.tolist()).squeeze()

print(len(sequence_list))
for vot_seq in vot_excluded_seqs:
    sequence_list.remove(vot_seq)
print(len(sequence_list))
data = []

for idx, video in tqdm(enumerate(sequence_list)):
    
    video_path = os.path.join(root, video)
    frames_list=[os.path.join(video_path, frame) for frame in os.listdir(video_path) if frame.endswith(".jpg") ]
    
    video_gt_path = os.path.join(video_path,'groundtruth.txt')
    occlusion_file = os.path.join(video_path, "absence.label")
    video_gt = printBB(root, video_path, video_gt_path)
    
    
    with open(occlusion_file, 'r', newline='') as f:
        occlusion = [int(v[0]) for v in csv.reader(f)]
        
    for frame_num, (frame, anno, occ) in enumerate(zip(sorted(frames_list),video_gt, occlusion)):
        img_path = frame
        # img = cv2.imread(img_path)
        
        img = Image.open(img_path)
        image_width,image_height  = img.size
        
        x = int(anno[0])
        y = int(anno[1])
        w = int(anno[2])
        h = int(anno[3])
        
        bbox_exist = 1 if occ==0 else 0
        bbox_border = int(x<=0 or y<=0 or (x+w)>=image_width-1 or (y+h)>=image_height-1)

        
        data.append([str(idx), video, frame_num, img_path, [x,y,w,h], [image_width, image_height], 'got10k',bbox_exist, bbox_border])
    # sanity save
    # df = pd.DataFrame(data, columns=["sequence_id","track_id","frame_index","img_path","bbox","frame_shape","dataset","presence","near_corner"])
    # df.to_csv("got10k.csv")

df = pd.DataFrame(data, columns=["sequence_id","track_id","frame_index","img_path","bbox","frame_shape","dataset","presence","near_corner"])
df.to_csv("got10k_vot_excluded.csv")
        
