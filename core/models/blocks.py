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
from timm.models.layers import trunc_normal_


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


class SepConv(nn.Module):
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



def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    
class Attention(nn.Module):
    """
    modified from
    src: https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/attention.py#L126
    """
        
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
    
    def forward(self, h_):
        q = self.q(h_)
        k = self.k(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
       
        return w_

class SpatialSelfCrossAttention(nn.Module):
    
    """
    attention module that outputs self and cross attention
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        
        self.attention = Attention(in_channels=in_channels)
        
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def proc_attention(self, v, w_, h):
        
        # attend to values
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)
        return h_
        
    def forward(self, x1, x2):
        
        # spatial self-attention
        h_1 = x1
        h_1 = self.norm(h_1)
        w_1 = self.attention(h_1)
        v = self.v(h_1)
        b,c,h,w = v.shape
        v = rearrange(v, 'b c h w -> b c (h w)')
        h_1 = self.proc_attention(v, torch.nn.functional.softmax(w_1, dim=2), h)
        
        # spatial cross-attention
        h_2 = x2
        h_2 = self.norm(h_2)
        w_2 = self.attention(h_2)
        h_12 = self.proc_attention(v, torch.nn.functional.softmax(w_1+w_2, dim=2), h)

        
        return x1+h_1, x1+h_12


       
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
        self.cls_dw = CorrelationConcat(num_channels=outchannels, conv_block=conv_block, gaussian_map=gaussian_map)
        self.reg_dw = CorrelationConcat(num_channels=outchannels, conv_block=conv_block, gaussian_map=gaussian_map)

        # box pred head
        for i in range(towernum):
            tower.append(SepConv(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            tower.append(nn.BatchNorm2d(outchannels))
            tower.append(nn.ReLU())

        # cls tower
        for i in range(towernum):
            cls_tower.append(SepConv(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            cls_tower.append(nn.BatchNorm2d(outchannels))
            cls_tower.append(nn.ReLU())

        self.add_module("bbox_tower", nn.Sequential(*tower))
        self.add_module("cls_tower", nn.Sequential(*cls_tower))

        # reg head
        self.bbox_pred = SepConv(outchannels, 4, kernel_size=3, stride=1, padding=1)
        self.cls_pred = SepConv(outchannels, 1, kernel_size=3, stride=1, padding=1)

        # adjust scale
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)))

    def forward(self, search, dynamic, kernel, update=None, gaussian_val=None):
        # encode first
        z = kernel.reshape(kernel.size(0), kernel.size(1), -1) if update is None else update.reshape(update.size(0), update.size(1), -1)
        cls_x = self.cls_encode(search)  # [z11, z12, z13]
        cls_d = self.cls_encode(dynamic)
        
        reg_x = self.reg_encode(search)  # [x11, x12, x13]
        reg_d = self.reg_encode(dynamic)
        
        # cls and reg DW
        cls_dw = self.cls_dw(z, cls_x, cls_d, gaussian_val)
        reg_dw = self.reg_dw(z, reg_x, reg_d, gaussian_val)
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
            SepConv(in_channels, out_channels, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.matrix11_s(x)
    
    
class CorrelationConcat(nn.Module):
    """
    Mobile Correlation module
    """

    def __init__(self, num_channels: int, num_corr_channels: int = 256, conv_block: str = "regular", gaussian_map=False):
        super().__init__()
        
        self.gaussian_map = gaussian_map
        in_size = num_channels + num_corr_channels 
        if self.gaussian_map:  in_size = in_size + 2 # 2 chan gaussian map, 1 for t-2 and 2 for t-1 
        
        self.enc = nn.Sequential(
            SepConv(in_size, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        self.weight = nn.Parameter(torch.empty(1, 2, 32, 32)) #gaussian map channels
        trunc_normal_(self.weight, std=.02)
        
    def forward(self, z, x, d, g=None):
        
        b, c, w, h = x.size()
        s = torch.matmul(z.permute(0, 2, 1), x.view(b, c, -1)).view(b, -1, w, h)
        s = torch.cat([s, d], dim=1) if g==None else torch.cat([s, d, g*self.weight], dim=1) # applying a broadcast weight factor to the gaussian_map parameter g (b, 2, 16, 16)
        s = self.enc(s)
        return s
    