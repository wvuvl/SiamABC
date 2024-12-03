import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
import json

def load_annotation_file_otb(ann_file):
    bboxes = []
    frame_num = 0
    with open(ann_file) as f:
        data = f.read().rstrip().split('\n')
        for bb in data:
            if ',' in bb: x1, y1, w, h = [int(i) for i in bb.split(',')]
            else: x1, y1, w, h = [int(float(i)) for i in bb.split()]
            #x1, y1, w, h
            bboxes.append([x1, y1, w, h]) #, frame_num])
            frame_num+=1

    return bboxes


video_dir = '/new_local_storage/zaveri/SOTA_Tracking_datasets/avist/sequences'
gt_dir = '/new_local_storage/zaveri/SOTA_Tracking_datasets/avist/anno/'
out_of_view = '/new_local_storage/zaveri/SOTA_Tracking_datasets/avist/out_of_view'
full_occlusions = '/new_local_storage/zaveri/SOTA_Tracking_datasets/avist/full_occlusion'

data = []
json_dict = {}
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
    
    image_files = [os.path.join(video, frame) for frame in frames if os.path.splitext(frame)[1]=='.jpg']
    gt_rect = gt
    
    min_num = min(len(image_files), len(gt_rect))
    
    json_dict[video] = {}
    json_dict[video]['video_dir'] = video
    json_dict[video]['img_names'] = image_files[:min_num]
    json_dict[video]['init_rect'] = gt_rect[0]
    json_dict[video]['gt_rect'] = gt_rect[:min_num]
    
    if len( json_dict[video]['img_names']) != len(json_dict[video]['gt_rect']): print("not equal")
    
    for anno_idx, anno in enumerate(gt):
        img_path = os.path.join(video_path, frames[anno_idx])
        img = Image.open(img_path)
        image_width,image_height  = img.size
        
        x = int(anno[0])
        y = int(anno[1])
        w = int(anno[2])
        h = int(anno[3])

        bbox_border = int(x<=0 or y<=0 or (x+w)>=image_width-1 or (y+h)>=image_height-1)
        bbox_exist = 0 if int(video_full_occlusions_frames[anno_idx]) == 1 or int(video_out_of_view_frames[anno_idx]) == 1 else 1
        
        # data.append([str(idx), video, anno[4], img_path, anno[:4], [image_width, image_height], 'avist',bbox_exist, bbox_border])

with open("AVIST.json", "w") as outfile: 
    json.dump(json_dict, outfile)
    
# df = pd.DataFrame(data, columns=["sequence_id","track_id","frame_index","img_path","bbox","frame_shape","dataset","presence","near_corner"])
# df.to_csv("AVIST.csv")
