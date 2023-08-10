import os
from tqdm import tqdm


otb_path = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/got10k/OTB'
out_path = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/got10k/OTB_bbox_dict.txt'

bbox_str = ''
for i in tqdm(os.listdir(otb_path)):

    ann_file = os.path.join(otb_path,i,'groundtruth_rect.txt')

    bboxes = []
    frame_num = 0
    with open(ann_file) as f:
        bb = f.read().rstrip().split('\n')[0]
        if ',' in bb: x1, y1, w, h = [int(i) for i in bb.split(',')]
        else: x1, y1, w, h = [int(float(i)) for i in bb.split()]

    bbox_str += f'{i}.txt,{x1},{y1},{w},{h}\n'
    
with open(out_path, 'a') as f:
    f.write(bbox_str)    
    