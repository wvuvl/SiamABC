"""

Modified from
Main Author of this file: FEAR
Repo: https://github.com/PinataFarms/FEARTracker/tree/main
File: https://github.com/PinataFarms/FEARTracker/blob/main/model_training/model/fear_net.py

"""

from typing import Dict, Tuple, Any, Tuple, List

import torch
import torch.nn as nn

# from blocks import Encoder, AdjustLayer, BoxTower, SpatialSelfCrossAttention
# TARGET_CLASSIFICATION_KEY = "TARGET_CLASSIFICATION_KEY"
# TARGET_REGRESSION_LABEL_KEY = "TARGET_REGRESSION_LABEL_KEY"
# SIMSIAM_SEARCH_OUT_KEY = "SIMSIAM_SEARCH_OUT_KEY"
# SIMSIAM_DYNAMIC_OUT_KEY = "SIMSIAM_DYNAMIC_OUT_KEY"


from models.blocks import Encoder, AdjustLayer, BoxTower, SpatialSelfCrossAttention
from utils.utils import make_grid
import constants 


class AEVTNet(nn.Module):
    def __init__(
        self,
        simsiam_dim: int = 2048,
        simsiam_pred_dim: int = 512,
        pretrained: bool = True,
        score_size: int = 25,
        adjust_channels: int = 256,
        total_stride: int = 8,
        instance_size: int = 255,
        towernum: int = 4,
        max_layer: int = 3,
        crop_template_features: bool = True,
        conv_block: str = "regular",
        gaussian_map: bool = False,
        **kwargs,
    ) -> None:
        max_layer2name = {3: "layer2", 4: "layer1"}
        assert max_layer in max_layer2name

        super().__init__()
        self.encoder = Encoder(pretrained)
        
        
        
        self.neck = AdjustLayer(
            in_channels=self.encoder.encoder_channels[max_layer2name[max_layer]], out_channels=adjust_channels
        )
        
        
        # SimSiam style MLP for correlation training
        # WARNING: when using batchnorm1d, make sure to use more than 1 batch size, it will fail otherwise
        # build a 3-layer projector
        # avgpooling similar to alexnet
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(nn.Linear(adjust_channels*6*6,simsiam_dim, bias=False),
                                        nn.BatchNorm1d(simsiam_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(simsiam_dim, simsiam_dim, bias=False),
                                        nn.BatchNorm1d(simsiam_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(simsiam_dim, simsiam_dim, bias=False),
                                        nn.BatchNorm1d(simsiam_dim, affine=False) # output layer
                                        )
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(simsiam_dim, simsiam_pred_dim, bias=False),
                                        nn.BatchNorm1d(simsiam_pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(simsiam_pred_dim, simsiam_dim) # output layer
                                        )   
        
        self.SpatialSelfCrossAttention = SpatialSelfCrossAttention(in_channels=adjust_channels)
        self.connect_model = BoxTower(
            inchannels=adjust_channels,
            outchannels=adjust_channels,
            towernum=towernum,
            conv_block=conv_block,
            gaussian_map=gaussian_map
        )
        
        self.score_size = score_size
        self.total_stride = total_stride
        self.instance_size = instance_size
        self.size = 1
        self.max_layer = max_layer
        self.crop_template_features = crop_template_features
        self.grid_x = torch.empty(0)
        self.grid_y = torch.empty(0)
        self.features = None
        # self.grids(self.size)

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.encoder.stages[: self.max_layer]:
            x = stage(x)
        return x

    def get_features(self, crop: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(crop)
        features = self.neck(features)
        return features        
        
    # def grids(self, size: int) -> None:
    #     """
    #     each element of feature map on input search image
    #     :return: H*W*2 (position for each element)
    #     """
    #     grid_x, grid_y = make_grid(self.score_size, self.total_stride, self.instance_size)
    #     self.grid_x, self.grid_y = grid_x.unsqueeze(0).repeat(size, 1, 1, 1), grid_y.unsqueeze(0).repeat(size, 1, 1, 1)

    def connector(self, template_features: torch.Tensor, self_attention_features: torch.Tensor, cross_attention_features: torch.Tensor, gaussian_val: None or torch.Tensor) -> Dict[str, torch.Tensor]:
        bbox_pred, cls_pred, _, _ = self.connect_model(self_attention_features, cross_attention_features, template_features, gaussian_val=gaussian_val)
        return bbox_pred, cls_pred

    def simsiam_forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        x1 = self.avgpool(x1)
        x1 = torch.flatten(x1, 1)
        x2 = self.avgpool(x2)
        x2 = torch.flatten(x2, 1)
        
        # compute features for one view classifier
        z1 = self.classifier(x1) # NxC
        z2 = self.classifier(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()
    
    
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None or torch.Tensor]) -> Dict[str, torch.Tensor or List[torch.Tensor]]:
        template, search, dynamic, gaussian_val = x
        template_features = self.get_features(template)
        search_features = self.get_features(search)
        dynamic_features = self. get_features(dynamic)
        self_attention_features, cross_attention_features = self.SpatialSelfCrossAttention(search_features, dynamic_features)
        bbox_pred, cls_pred =  self.connector(template_features=template_features, self_attention_features=self_attention_features, cross_attention_features=cross_attention_features, gaussian_val=gaussian_val)
        
        simsiam_out_search = self.simsiam_forward(template_features, search_features)
        simsiam_out_dynamic = self.simsiam_forward(template_features, dynamic_features)
        
        return {
            constants.TARGET_REGRESSION_LABEL_KEY: bbox_pred,
            constants.TARGET_CLASSIFICATION_KEY: cls_pred,
            constants.SIMSIAM_SEARCH_OUT_KEY: simsiam_out_search,
            constants.SIMSIAM_DYNAMIC_OUT_KEY: simsiam_out_dynamic
        }

    def track(
        self,
        search: torch.Tensor,
        dynamic: torch.Tensor,
        template_features: torch.Tensor,
        gaussian_val: None or torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        search_features = self.get_features(search)
        dynamic_features = self. get_features(dynamic)
        self_attention_features, cross_attention_features = self.SpatialSelfCrossAttention(search_features, dynamic_features)
        bbox_pred, cls_pred =  self.connector(template_features=template_features, self_attention_features=self_attention_features, cross_attention_features=cross_attention_features, gaussian_val=gaussian_val)
        
        return {
            constants.TARGET_REGRESSION_LABEL_KEY: bbox_pred,
            constants.TARGET_CLASSIFICATION_KEY: cls_pred,
        }
        
        
if __name__ == '__main__':
    model = AEVTNet(gaussian_map=True)
    print()
    search = torch.randn((2,3,256,256))
    dynamic = torch.randn((2,3,256,256))
    template = torch.randn((2,3,128,128))
    template_features = model.get_features(template)
    search_features = model.get_features(search)
    dynamic_features = model. get_features(dynamic)
    gaussian_val = torch.randn((2,2,32,32))
    self_attention_features, cross_attention_features = model.SpatialSelfCrossAttention(search_features, dynamic_features)
    bbox_pred, cls_pred,_,_ =  model.connect_model(self_attention_features, cross_attention_features, template_features, gaussian_val=gaussian_val)
    
    simsiam_out_search = model.simsiam_forward(template_features, search_features)
    simsiam_out_dynamic = model.simsiam_forward(template_features, dynamic_features)
        
    print(bbox_pred, cls_pred)