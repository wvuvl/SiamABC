import random
from abc import ABC
from math import ceil
from typing import Any, Dict

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

    
    def _read_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.data_path)
        negative = data[data["presence"] == 0]
        negative_ratio = len(negative) / len(data)
        num_neg_samples_to_keep = max(0, int(min(negative_ratio, self.negative_ratio) * len(data)))
        num_neg_samples_to_drop = len(negative) - num_neg_samples_to_keep
        dropped_negatives = np.random.choice(negative.index, num_neg_samples_to_drop, replace=False)
        data = data.drop(dropped_negatives)
        data = data.reset_index(drop=True)
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
        
        search_index = random.choice(track_indices)
        search_item = self.data.iloc[search_index]
        
        
        search_items = self.data.iloc[track_indices]
        
        # dynamic_items = (
        #     search_items[
        #         (search_items["frame_index"] <= search_item["frame_index"])
        #         & (search_items["presence"] == 1) 
        #     ]
        #     .sort_values(by='frame_index', ascending=False)
        # )
        # dynamic_item =  dynamic_items.iloc[:2].sample(1).iloc[0] if len(dynamic_items) > 1 else dynamic_items.iloc[0]

        
        # prev_dynamic_items = (
        #     search_items[
        #         (search_items["frame_index"] <= dynamic_item["frame_index"])
        #         & (search_items["presence"] == 1) 
        #     ]
        #     .sort_values(by='frame_index', ascending=False)
        # )
        
        # prev_dynamic_item =  prev_dynamic_items.iloc[1] if len(prev_dynamic_items) > 1 else prev_dynamic_items.iloc[0]

        dynamic_item = (
            search_items[
                (search_items["frame_index"] > search_item["frame_index"] - self.frame_offset/2) # if frame_offset == 70, it is only going to search for 35 frames since we also need to account for the previous dynamic frame 
                & (search_items["frame_index"] <= search_item["frame_index"])
                & (search_items["presence"] == 1) 
            ]
            .sample(1)
            .iloc[0]
        )
        
        prev_dynamic_item = (
            search_items[
                (search_items["frame_index"] > dynamic_item["frame_index"] - self.frame_offset/2)
                & (search_items["frame_index"] <= dynamic_item["frame_index"])
                & (search_items["presence"] == 1) 
            ]
            .sample(1)
            .iloc[0]
        )
            
        
        return dict(template=template_item, search=search_item, dynamic=dynamic_item, prev_dynamic=prev_dynamic_item)

if __name__ == '__main__':
    data_path = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/code/AEVT/core/dataset_utils/AVIST.csv'
    sampler = TrackSampler(data_path,negative_ratio=1.,frame_offset=70,num_samples=20000)
    sampler.parse_samples()
    samples = sampler.extract_sample(12)