"""

Modified from
Main Author of this file: FEAR
Repo: https://github.com/PinataFarms/FEARTracker/tree/main
File: https://github.com/PinataFarms/FEARTracker/blob/main/model_training/model/blocks.py

"""

from typing import Any, Union, Tuple, List

import torch
import torch.nn as nn
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from einops import rearrange
from torch.nn import init
from torchvision.models.resnet import resnet50
from torchvision.models import regnet_x_8gf
from core.models import neuron
import torch.nn.functional as F

def conv_2d(inp, oup, kernel_size=3, stride=1, padding=0, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.SiLU())
    return conv

class FastParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=256, squeeze=2):
        super().__init__()
        
        self.squeeze=squeeze
        self.wv=nn.Conv2d(channel,channel//self.squeeze,kernel_size=(1,1))
        
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//self.squeeze,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wq=nn.Conv2d(channel,channel//self.squeeze,kernel_size=(1,1))
        # self.sp_wz=nn.Conv2d(1,1,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, x1):
        b, c, h, w = x1.size()
        
        
        
        wv=self.wv(x1) #bs,c//2,h,w
        wv=wv.reshape(b,c//self.squeeze,-1) #bs,c//2,h*w
        
        #Channel-only Self-Attention
        channel_wq=self.ch_wq(x1) #bs,1,h,w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.sum(wv*channel_wq.permute(0,2,1),dim=2).unsqueeze(-1).unsqueeze(-1)
        channel_weight=self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1)).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1

        #Spatial-only Self-Attention
        spatial_wq=self.sp_wq(x1) #bs,c,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c,1,1
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//self.squeeze) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.sum(spatial_wq.permute(0,2,1)*wv,dim=1).unsqueeze(1)
        spatial_weight=spatial_wz.reshape(b,1,h,w) #bs,1,h,w
        # spatial_weight=self.sp_wz(spatial_weight)
        
        out=(self.sigmoid(channel_weight)+self.sigmoid(spatial_weight))*x1
        
        return out
    
class EncoderResNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.last_layer_channels = 1024
        self.model = self._load_model()
        self.layers = self._get_layers()
        

    def _load_model(self) -> Any:
        model = resnet50(pretrained=self.pretrained)
        return model

    def _get_layers(self) -> List[Any]:
        layers = [
            self.model.conv1,
            self.model.bn1, 
            self.model.relu, 
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            # self.model.layer4,
        ]
        return layers

    def forward(self, x: Any) -> List[Any]:
        encoder_maps = []
        for layer in self.layers:
            x = layer(x)
            encoder_maps.append(x)
        return encoder_maps
    
class Encoder(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.model = self._load_model()
        self.stages = self._get_stages()
        self.encoder_channels = {
           "layer0": 352,
           "layer1": 112,
           "layer2": 32,
           "layer3": 24,
           "layer4": 16,
        }

    def _load_model(self) -> Any:
        model_name = "fbnet_c"
        model = fbnet(model_name, pretrained=self.pretrained)
        return model

    def _get_stages(self) -> List[Any]:
        stages = [
            self.model.backbone.stages[:2],
            self.model.backbone.stages[2:5],
            self.model.backbone.stages[5:9],
            self.model.backbone.stages[9:18],
            self.model.backbone.stages[18:23],
        ]
        return stages

    def forward(self, x: Any) -> List[Any]:
        encoder_maps = []
        for stage in self.stages:
            x = stage(x)
            encoder_maps.append(x)
        return encoder_maps


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class AdjustLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, crop_rate: int = 4):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.size_threshold = 20
        self.crop_rate = crop_rate

    def forward(self, x):
        x_ori = self.downsample(x)
        adjust = x_ori
        return adjust

class BoxTower(nn.Module):
    """
    Box Tower for FCOS regression
    """
    def __init__(
        self,
        towernum: int = 4,
        conv_block: str = "regular",
        inchannels: int = 512,
        outchannels: int = 256,
        gaussian_map=False
    ):
        super().__init__()
        tower = []
        cls_tower = []
        # encode backbone
        self.cls_encode = EncodeBackbone(in_channels=inchannels, out_channels=outchannels, conv_block=conv_block)
        self.reg_encode = EncodeBackbone(in_channels=inchannels, out_channels=outchannels, conv_block=conv_block)
        
        # self.cls_dw = Correlation2xConcat(num_channels=outchannels, conv_block=conv_block)
        # self.reg_dw = Correlation2xConcat(num_channels=outchannels, conv_block=conv_block)
        
        self.cls_dw = CorrelationConcat(num_channels=outchannels)
        self.reg_dw = CorrelationConcat(num_channels=outchannels)
        
        # box pred head
        for i in range(4):
            tower.append(ConvBlock(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            tower.append(nn.BatchNorm2d(outchannels))
            tower.append(nn.ReLU())
        # cls tower
        for i in range(towernum):
            cls_tower.append(ConvBlock(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            cls_tower.append(nn.BatchNorm2d(outchannels))
            cls_tower.append(nn.ReLU())

        self.add_module("bbox_tower", nn.Sequential(*tower))
        self.add_module("cls_tower", nn.Sequential(*cls_tower))

        # reg head
        self.bbox_pred = ConvBlock(outchannels, 4, kernel_size=3, stride=1, padding=1)
        self.cls_pred = ConvBlock(outchannels, 1, kernel_size=3, stride=1, padding=1)

        # adjust scale
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)))

    def forward(self, search_org, search, kernel): #forward(self, search, dynamic, kernel):
        
        # encode first
        cls_z = kernel.reshape(kernel.size(0), kernel.size(1), -1)
        cls_x = self.cls_encode(search)
        
        reg_z = kernel.reshape(kernel.size(0), kernel.size(1), -1)
        reg_x = self.reg_encode(search) 
        
        # cls and reg DW
        cls_dw = self.cls_dw(cls_z, cls_x, search_org )
        reg_dw = self.reg_dw(reg_z, reg_x, search_org) 
        x_reg = self.bbox_tower(reg_dw)
        x = self.adjust * self.bbox_pred(x_reg) + self.bias
        x = torch.exp(x)

        # cls tower
        c = self.cls_tower(cls_dw)
        cls = 0.1 * self.cls_pred(c)

        return x, cls, cls_dw, x_reg

class EncodeBackbone(nn.Module):
    """
    Encode backbone feature
    """

    def __init__(self, in_channels, out_channels, conv_block: str = "regular"):
        super().__init__()
        self.matrix11_s = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.matrix11_s(x)
    
    
class CorrelationConcat(nn.Module):
    """
    Correlation module
    """

    def __init__(self, num_channels: int, num_corr_channels: int = 64):
        super().__init__()
        

        in_size = num_channels + num_corr_channels             
        self.enc = nn.Sequential(
            ConvBlock(in_size, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, z, x, d):
        
        b, c, w, h = x.size()
        s = torch.matmul(z.permute(0, 2, 1), x.view(b, c, -1)).view(b, -1, w, h)
        s = torch.cat([s, d], dim=1)
        s = self.enc(s)
        return s   