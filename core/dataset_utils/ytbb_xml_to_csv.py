import xml.etree.ElementTree as ET
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

anno_path = '/new_local_storage/zaveri/SOTA_Tracking_datasets/ytbb_decoded/youtubebbdevkit2017/youtubebb2017/Annotations'
jpg_path = '/new_local_storage/zaveri/SOTA_Tracking_datasets/ytbb_decoded/youtubebbdevkit2017/youtubebb2017/JPEGImages'
save_file = 'ytbb.csv'

file = '/new_local_storage/zaveri/SOTA_Tracking_datasets/ytbb_decoded/youtubebbdevkit2017/youtubebb2017/ImageSets/Main/trainval.txt'
lst = np.array(pd.read_csv(file, header=None).values.tolist()).squeeze()
lst.sort()

anno_dict = {}

for anno_file in tqdm(sorted(os.listdir(anno_path))):
    

    xmlTree = ET.parse(os.path.join(anno_path,anno_file))

    elemList = {}

    for elem in xmlTree.iter():
        elemList[elem.tag] = elem.text


    total_ID = elemList['total_ID']

    filename = elemList['filename']
    
    if os.path.exists( os.path.join(jpg_path,filename))==False: 
        print('JPG does not exist')
        break
    # else:
    #     print('JPG does exist')
    
    width = int(elemList['width'])
    height = int(elemList['height'])

    name = elemList['name']

    xmin = int(elemList['xmin'])
    ymin = int(elemList['ymin'])
    xmax = int(elemList['xmax'])
    ymax = int(elemList['ymax'])

    x = xmin
    y = ymin
    w = xmax-xmin
    h = ymax-ymin
    
    bbox_exist = int((w > 0) & (h > 0))
    bbox_border = int(x<=0 or y<=0 or (x+w)>=width-1 or (y+h)>=height-1)
                
    if total_ID not in list(anno_dict.keys()): 
        anno_dict[total_ID] = {}
    
    anno_dict[total_ID][filename]={}
    anno_dict[total_ID][filename]['filename'] = os.path.join(jpg_path,filename)
    anno_dict[total_ID][filename]['shape'] = [width,height]
    anno_dict[total_ID][filename]['name'] = name
    anno_dict[total_ID][filename]['bbox'] = [x,y,w,h]
    anno_dict[total_ID][filename]['bbox_exist'] = bbox_exist
    anno_dict[total_ID][filename]['bbox_border'] = bbox_border
                

keys = list(anno_dict.keys())

data = []

for sequence_id, total_ID_key in enumerate(tqdm(keys)):
    
    total_ID = anno_dict[total_ID_key]
    file_keys = list(total_ID.keys())
    for frame, file_key in enumerate(file_keys):
        
        filename = anno_dict[total_ID_key][file_key]['filename']
        shape = anno_dict[total_ID_key][file_key]['shape']
        object_name = anno_dict[total_ID_key][file_key]['name']
        bbox = anno_dict[total_ID_key][file_key]['bbox']
        bbox_exist = anno_dict[total_ID_key][file_key]['bbox_exist']
        bbox_border = anno_dict[total_ID_key][file_key]['bbox_border']

        data.append([str(sequence_id),sequence_id, frame, filename, bbox, shape, 'ytbb', bbox_exist, bbox_border])

df = pd.DataFrame(data, columns=["sequence_id","track_id","frame_index","img_path","bbox","frame_shape","dataset","presence","near_corner"])
df.to_csv(save_file)