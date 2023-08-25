import random
from abc import ABC
from math import ceil
from typing import Any, Dict
from tqdm import trange, tqdm
import numpy as np
import pandas as pd


class TrackSampler(ABC):
    def __init__(
        self,
        data_path: str,
        negative_ratio: float,
        frame_offset: int,
        num_samples: int
    ):  
        self.data_path = data_path
        self.negative_ratio = negative_ratio
        self.frame_offset = frame_offset
        self.num_samples = num_samples
        self.data = None
        self.mapping = None
        
        self.epoch_data = None
        self.template_data = None
        self.num_tracks = None


    def __len__(self) -> int:
        return len(self.epoch_data)

    
    def drop_bad_bboxes(self, data):
        bboxes = data["bbox"]
        frame_shape = data["frame_shape"]
        indexes = []
        for idx, (bbox, i_frame_shape) in tqdm(enumerate(zip(bboxes, frame_shape)), desc="filtering data - "):
            x,y,w,h = eval(bbox)
            im_w,im_h = eval(i_frame_shape)
            if w <= 3 or h <= 3 or x >= im_w-3 or y >= im_h-3 or x<0 or y<0:
                indexes.append(idx)
        data = data.drop(indexes)
        data = data.reset_index(drop=True)
        return data 
        
    def _read_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.data_path)
        negative = data[data["presence"] == 0]
        negative_ratio = len(negative) / len(data)
        num_neg_samples_to_keep = max(0, int(min(negative_ratio, self.negative_ratio) * len(data)))
        num_neg_samples_to_drop = len(negative) - num_neg_samples_to_keep
        dropped_negatives = np.random.choice(negative.index, num_neg_samples_to_drop, replace=False)
        data = data.drop(dropped_negatives)
        data = data.reset_index(drop=True)
        data = self.drop_bad_bboxes(data)
        return data

    def resample(self) -> None:
        if self.num_tracks == len(self.template_data):
            self.epoch_data = self.template_data.sample(self.num_samples).reset_index(drop=True)
        else:
            self.epoch_data = (
                self.template_data.groupby("track_id")
                .sample(int(ceil((self.num_samples / self.num_tracks))), replace=True)
                .sample(self.num_samples)
                .reset_index(drop=True)
            )

    # gets rid of the frames where the object does not exist or in the corner
    def parse_samples(self) -> None:
        self.data = self._read_data()
        self.template_data = self.data[(self.data["presence"] == 1) & (~self.data["near_corner"])]
        self.num_tracks = len(self.template_data["track_id"].unique())
        self.mapping = self.data.groupby(["track_id"]).groups
        self.resample()

    
    """
    
    extracting 4 samples,
    template: any
    search: current search region, anywhere in the frame
    dynamic: frame before search region
    prev_dynamic: frame before dynamic region
    
    """
    def extract_sample(self, idx: int) -> Dict[str, Any]:
        template_item = self.epoch_data.iloc[idx]
        track_indices = self.mapping[template_item["track_id"]]
                
        search_items = self.data.iloc[track_indices]
        
        # negatives allowed
        search_item = (search_items.sample(1).iloc[0]) #(search_items[(search_items["presence"] == 1)].sample(1).iloc[0])

        
        dynamic_item = (
            search_items[
                (search_items["frame_index"] > search_item["frame_index"] - 15) #- self.frame_offset/4) # if frame_offset == 70, it is only going to search for 35 frames since we also need to account for the previous dynamic frame 
                & (search_items["frame_index"] <= search_item["frame_index"])
                & (search_items["presence"] == 1) 
            ]
            .sample(1)
            .iloc[0]
        )
        
        prev_dynamic_item = (
            search_items[
                (search_items["frame_index"] > dynamic_item["frame_index"] - 15) #- self.frame_offset/4)
                & (search_items["frame_index"] <= dynamic_item["frame_index"] )
                & (search_items["presence"] == 1) 
            ]
            .sample(1)
            .iloc[0]
        )
            
        
        return dict(template=template_item, search=search_item, dynamic=dynamic_item, prev_dynamic=prev_dynamic_item)

if __name__ == '__main__':
    import time
    # from utils import read_img
    

    data_path = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/code/AEVT/AVIST.csv'
    sampler = TrackSampler(data_path,negative_ratio=1.,frame_offset=70,num_samples=20000)
    sampler.parse_samples()
    # samples = sampler.extract_sample(12)
    start = time.time()
    for i in trange(78000):
        samples = sampler.extract_sample(12)
        # template_image = read_img(samples["template"]["img_path"])
        # search_image = read_img(samples["search"]["img_path"])
        # dynamic_image = read_img(samples["dynamic"]["img_path"])
        # prev_dynamic_image = read_img(samples["prev_dynamic"]["img_path"])
    samples = sampler.data["img_path"]    
    # for i in tqdm(samples):
    #     prev_dynamic_image = read_img(i)    
    print(time.time()-start)
