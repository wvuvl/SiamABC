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

from tqdm import tqdm
from got10k.datasets import DTB70

root_dir = '/new_local_storage/zaveri/SOTA_Tracking_datasets/DTB70/'
dataset = DTB70(root_dir)


json_dict = {}

for index, (img_files, anno) in tqdm(enumerate(dataset)):
    
    img_files = [i.replace(root_dir,'') for i in img_files]
    video = dataset.seq_names[index]
    json_dict[video] = {}
    json_dict[video]['video_dir'] = video
    json_dict[video]['img_names'] = img_files
    json_dict[video]['init_rect'] = anno[0].tolist()
    json_dict[video]['gt_rect'] = anno.tolist()
    

print(len(json_dict.keys()))
with open("DTB70.json", "w") as outfile: 
    json.dump(json_dict, outfile)