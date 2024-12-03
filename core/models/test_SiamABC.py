"""

Modified from
Main Author of this file: FEAR
Repo: https://github.com/PinataFarms/FEARTracker/tree/main
File: https://github.com/PinataFarms/FEARTracker/blob/main/model_training/model/fear_net.py

"""

from typing import Dict, Tuple, Any, Tuple, List

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
    

    
# from blocks import Encoder, AdjustLayer, BoxTower, SpatialSelfCrossAttention
# TARGET_CLASSIFICATION_KEY = "TARGET_CLASSIFICATION_KEY"
# TARGET_REGRESSION_LABEL_KEY = "TARGET_REGRESSION_LABEL_KEY"
# SIMSIAM_SEARCH_OUT_KEY = "SIMSIAM_SEARCH_OUT_KEY"
# SIMSIAM_DYNAMIC_OUT_KEY = "SIMSIAM_DYNAMIC_OUT_KEY"


from blocks import Encoder, EncoderRegNet, AdjustLayer, BoxTower, SpatialSelfCrossAttention, Correlation, EncoderResNet
from resnet import resnet50


class SiamABCNet(nn.Module):
    def __init__(
        self,
        simsiam_dim: int = 2048,
        simsiam_pred_dim: int = 512,
        pretrained: bool = True,
        adjust_channels: int = 256,
        towernum: int = 2,
        max_layer: int = 4,
        conv_block: str = "regular",
        backbone = 'resnet',
        model_size = 'S',
        **kwargs,
    ):
        max_layer2name = {3: "layer2", 4: "layer1"}
        assert max_layer in max_layer2name

        super().__init__()
                
        # self.max_layer = max_layer
        # base_encoder = Encoder(pretrained)
        # self.encoder = nn.Sequential(*base_encoder.stages[:self.max_layer]) 
        # adjust_in_channels = base_encoder.encoder_channels[max_layer2name[max_layer]]

        
        if model_size=='S':
            self.max_layer = max_layer
            base_encoder = Encoder(pretrained)
            self.encoder = nn.Sequential(*base_encoder.stages[:self.max_layer]) 
            adjust_in_channels = base_encoder.encoder_channels[max_layer2name[max_layer]]
        elif model_size=='M':
            base_encoder = EncoderResNet(pretrained=pretrained) 
            adjust_in_channels = base_encoder.last_layer_channels            
            self.encoder = base_encoder #nn.Sequential(*base_encoder.layers)
        elif model_size=='L':
            base_encoder = resnet50(pretrained=True, last_layer = 'layer4') #EncoderResNet(pretrained=pretrained) 
            adjust_in_channels = 2048 #base_encoder.last_layer_channels            
            self.encoder = base_encoder #nn.Sequential(*base_encoder.layers)
            print("-- USING CUSTOM RESNET -- ")
        else:
            base_encoder = EncoderRegNet(pretrained=pretrained) 
            adjust_in_channels = base_encoder.last_layer_channels
            self.encoder = nn.Sequential(*base_encoder.layers)
        
        
        self.neck = AdjustLayer(in_channels=adjust_in_channels, out_channels=adjust_channels)
        
        
        self.SpatialSelfCrossAttention = SpatialSelfCrossAttention(in_channels=adjust_channels)
        
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
        
        
        
        
        self.connect_model = BoxTower(
            inchannels=adjust_channels,
            outchannels=adjust_channels,
            towernum=towernum,
            conv_block=conv_block
        )
        # self.correlation_block = Correlation(num_channels=adjust_channels)
        self.similarity = nn.CosineSimilarity(dim=1)
        self.similarity_avgpool = nn.AdaptiveAvgPool2d((16, 16))
        # self.sim_maxpool = nn.MaxPool2d((2,2),(2,2)) 
        
    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x

    def get_features(self, crop: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(crop)
        features = self.neck(features)
        return features        

    def connector(self, template_features: torch.Tensor, self_attention_features: torch.Tensor, cross_attention_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        bbox_pred, cls_pred, _, _ = self.connect_model(self_attention_features, cross_attention_features, template_features)
        return bbox_pred, cls_pred
    
    def calc_sim(self, x1, x2):
        
        x1 = self.similarity_avgpool(x1) #256x8x8 -> 256x16x16
        x1 = torch.flatten(x1, 1)
        # x2 = self.sim_maxpool(x2) #256x16x16 -> 256x8x8
        x2 = torch.flatten(x2, 1)
        
        return self.similarity(x1, x2).mean()
        
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
    
    
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor or List[torch.Tensor]]:
        template, dynamic_template, search, dynamic_search = x
        
        template_features = self.get_features(template)
        dynamic_template_features = self.get_features(dynamic_template)
        
        
        self_attention_template_features, cross_attention_template_features =  self.SpatialSelfCrossAttention(dynamic_template_features, template_features)
        
        search_features = self.get_features(search)
        dynamic_search_features = self. get_features(dynamic_search)
        self_attention_search_features, cross_attention_search_features = self.SpatialSelfCrossAttention(search_features, dynamic_search_features)
        

        bbox_pred, cls_pred =  self.connector(template_features=cross_attention_template_features, self_attention_features=cross_attention_search_features, cross_attention_features=self_attention_search_features)
        
        simsiam_out_search = self.simsiam_forward(cross_attention_template_features, self_attention_search_features)
        simsiam_out_dynamic = self.simsiam_forward(cross_attention_template_features, cross_attention_search_features)
        


    def track(
        self,
        search_features: torch.Tensor,
        dynamic_search_features: torch.Tensor,
        template_features: torch.Tensor,
        dynamic_template_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        
        self_attention_template_features, cross_attention_template_features = self.SpatialSelfCrossAttention(dynamic_template_features, template_features)

        self_attention_search_features, cross_attention_search_features = self.SpatialSelfCrossAttention(search_features, dynamic_search_features)

        bbox_pred, cls_pred =  self.connector(template_features=cross_attention_template_features, self_attention_features=cross_attention_search_features, cross_attention_features=self_attention_search_features)
        
        sim_score = self.calc_sim(cross_attention_template_features, search_features*cls_pred) #cross_attention_search_features*cls_pred) #cross_attention_search_features*cls_pred)
        

        


if __name__ == '__main__':
    from tqdm import trange
    from PIL import Image
    import numpy as np
    
    model = SiamABCNet(model_size='S')
    
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])


    # model_parameters = filter(lambda p: p.requires_grad, model.classifier.parameters())
    # cls_params = sum([np.prod(p.size()) for p in model_parameters])
    
    # model_parameters = filter(lambda p: p.requires_grad, model.predictor.parameters())
    # pred_params = sum([np.prod(p.size()) for p in model_parameters])

    # print(params-cls_params-pred_params)
    
    search_init = torch.randn((1,3,256,256))
    dynamic_search = torch.randn((1,3,256,256))
    template = torch.randn((1,3,128,128))
    dynamic_template = torch.randn((1,3,128,128))
    
    
    flops = FlopCountAnalysis(model.encoder, (template))
    encoder_temp_flops = flops.total()/ (1024*1024*1024)
    print(encoder_temp_flops)
    
    flops = FlopCountAnalysis(model.encoder, (dynamic_search))
    encoder_search_flops = flops.total()/ (1024*1024*1024)
    print(encoder_search_flops)
    
    print(encoder_temp_flops+encoder_search_flops)
    
    
    dynamic_search = model.get_features(dynamic_search)
    template = model.get_features(template)
    dynamic_template = model.get_features(dynamic_template)

    
    flops = FlopCountAnalysis(model.SpatialSelfCrossAttention, (template, dynamic_template))
    temp_sca_flops = flops.total()/ (1024*1024*1024)
    print(temp_sca_flops)
    
    flops = FlopCountAnalysis(model.SpatialSelfCrossAttention, (dynamic_search, dynamic_search))
    search_sca_flops = flops.total()/ (1024*1024*1024)
    print(search_sca_flops)
    
    flops = FlopCountAnalysis(model.connect_model, (dynamic_search, dynamic_search, template))
    reg_flops = flops.total()/ (1024*1024*1024)
    print(reg_flops)
    
    print(encoder_temp_flops+encoder_search_flops+temp_sca_flops+search_sca_flops+reg_flops)
    
    
    # import torchvision.models as models
    # from ptflops import get_model_complexity_info
    # import re

    # macs, params = get_model_complexity_info(model, (3, 128, 128), as_strings=True,
    # print_per_layer_stat=True, verbose=True)
    # # Extract the numerical value
    # flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
    # # Extract the unit
    # flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]

    # print('Computational complexity: {:<8}'.format(macs))
    # print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
    # print('Number of parameters: {:<8}'.format(params))

    

    # for i in trange(300000):
    #     img = np.asarray(Image.open('/new_local_storage/zaveri/code/SiamABC/temp.jpg'))
    #     search = model.get_features(search_init)
    #     model.track(search,dynamic_search, template, dynamic_template)