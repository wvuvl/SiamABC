import os
import cv2
from pycocotools.coco import COCO
import pandas as pd
from tqdm import tqdm

def get_class_list(cats):
    class_list = []
    for cat_id in cats.keys():
        class_list.append(cats[cat_id]['name'])
    return class_list


def _get_sequence_list(coco_set):
    ann_list = list(coco_set.anns.keys())
    seq_list = [a for a in ann_list if coco_set.anns[a]['iscrowd'] == 0]

    return seq_list

root = '/data/zaveri/SOTA_Tracking_datasets/coco2017'

split = 'train'
version = '2017'
img_pth = os.path.join(root, '{}{}/'.format(split, version))
anno_path = os.path.join(root, 'annotations/instances_{}{}.json'.format(split, version))

coco_set = COCO(anno_path)
cats = coco_set.cats


    

class_list = get_class_list(cats)
sequence_list = _get_sequence_list(coco_set)

data = []

for idx, seq in tqdm(enumerate(sequence_list)):
    class_name = cats[coco_set.anns[seq]['category_id']]['name']
    anno = coco_set.anns[seq]['bbox']
    path = coco_set.loadImgs([coco_set.anns[seq]['image_id']])[0]['file_name']
    img_path = os.path.join(img_pth, path)
    img = cv2.imread(img_path)
    
    x = int(anno[0])
    y = int(anno[1])
    w = int(anno[2])
    h = int(anno[3])
    
    bbox_exist = 1
    bbox_border = int(x<=0 or y<=0 or (x+w)>=img.shape[1]-1 or (y+h)>=img.shape[0]-1)
    
    data.append([str(idx), str(idx), 0, img_path, [x,y,w,h], [img.shape[1], img.shape[0]], 'coco2017',bbox_exist, bbox_border])
    
df = pd.DataFrame(data, columns=["sequence_id","track_id","frame_index","img_path","bbox","frame_shape","dataset","presence","near_corner"])
df.to_csv("coco2017.csv")
