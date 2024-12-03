import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

otb_path = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/got10k/OTB'
out_path = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/got10k/OTB_mp4_v2'
if os.path.exists(out_path) == False: os.makedirs(out_path)

for i in tqdm(os.listdir(otb_path)):
    
    vid_path = os.path.join(otb_path,i,'img','*.jpg')

    img_array = []
    for filename in sorted(glob.glob(vid_path)):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)


    out = cv2.VideoWriter(os.path.join(out_path,i+'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

