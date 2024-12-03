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
from torch.nn import init
from torchvision.models.resnet import resnet50
from torchvision.models import regnet_x_8gf
from core.models import neuron
from core.models import spiking_resnet
import torch.nn.functional as F

def conv_2d(inp, oup, kernel_size=3, stride=1, padding=0, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.SiLU())
    return conv

class LinearSelfAttention(nn.Module):
    def __init__(self, embed_dim, attn_dropout=0):
        super().__init__()
        self.qkv_proj = conv_2d(embed_dim, 1+2*embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = conv_2d(embed_dim, embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        self.embed_dim = embed_dim

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1)
        context_score = F.softmax(q, dim=-1)
        context_score = self.attn_dropout(context_score)

        context_vector = k * context_score
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        out = F.relu(v) * context_vector.expand_as(v)
        out = self.out_proj(out)
        return out

class LinearAttnFFN(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, dropout=0, attn_dropout=0):
        super().__init__()
        self.pre_norm_attn = nn.Sequential(
            nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1),
            LinearSelfAttention(embed_dim, attn_dropout),
            nn.Dropout(dropout)
        )
        self.pre_norm_ffn = nn.Sequential(
            nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1),
            conv_2d(embed_dim, ffn_latent_dim, kernel_size=1, stride=1, bias=True, norm=False, act=True),
            nn.Dropout(dropout),
            conv_2d(ffn_latent_dim, embed_dim, kernel_size=1, stride=1, bias=True, norm=False, act=False),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # self attention
        x = x + self.pre_norm_attn(x)
        # Feed Forward network
        x = x + self.pre_norm_ffn(x)
        return x
    
class MobileViTBlockv3_v2(nn.Module):
    def __init__(self, inp, attn_dim, ffn_multiplier, attn_blocks, patch_size):
        super(MobileViTBlockv3_v2, self).__init__()
        self.patch_h, self.patch_w = patch_size

        # local representation
        self.local_rep = nn.Sequential()
        self.local_rep.add_module('conv_3x3', conv_2d(inp, inp, kernel_size=3, stride=1, padding=1, groups=inp))
        self.local_rep.add_module('conv_1x1', conv_2d(inp, attn_dim, kernel_size=1, stride=1, norm=False, act=False))
        
        # global representation
        self.global_rep = nn.Sequential()
        ffn_dims = [int((ffn_multiplier*attn_dim)//16*16)] * attn_blocks
        for i in range(attn_blocks):
            ffn_dim = ffn_dims[i]
            self.global_rep.add_module(f'LinearAttnFFN_{i}', LinearAttnFFN(attn_dim, ffn_dim))
        self.global_rep.add_module('LayerNorm2D', nn.GroupNorm(num_channels=attn_dim, eps=1e-5, affine=True, num_groups=1))

        self.conv_proj = conv_2d(2*attn_dim, inp, kernel_size=1, stride=1, padding=0, act=False)

    def unfolding_pytorch(self, feature_map):
        batch_size, in_channels, img_h, img_w = feature_map.shape
        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )
        return patches, (img_h, img_w)

    def folding_pytorch(self, patches, output_size):
        batch_size, in_dim, patch_size, n_patches = patches.shape
        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        return feature_map

    def forward(self, x):
        res = x.clone()
        fm_conv = self.local_rep(x)
        x, output_size = self.unfolding_pytorch(fm_conv)
        x = self.global_rep(x)
        x = self.folding_pytorch(patches=x, output_size=output_size)
        x = self.conv_proj(torch.cat((x, fm_conv), dim=1))
        x = x + res
        return x
    

    
class ParallelPolarizedCrossAttention(nn.Module):

    def __init__(self, channel=256, squeeze=2):
        super().__init__()
        self.squeeze=squeeze
        self.ch_wv=ConvBlock(channel,channel//self.squeeze,kernel_size=(1,1))
        self.ch_wq=ConvBlock(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=ConvBlock(channel//self.squeeze,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=ConvBlock(channel,channel//self.squeeze,kernel_size=(1,1))
        self.sp_wq=ConvBlock(channel,channel//self.squeeze,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x1, x2):
        b, c, h, w = x1.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x1) #bs,c//2,h,w
        channel_wv=channel_wv.reshape(b,c//self.squeeze,-1) #bs,c//2,h*w
        
        channel_wq=self.ch_wq(x2) #bs,1,h,w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x1

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x1) #bs,c//2,h,w
        spatial_wv=spatial_wv.reshape(b,c//self.squeeze,-1) #bs,c//2,h*w
        
        spatial_wq=self.sp_wq(x2) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//self.squeeze) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x1
        
        
        out=spatial_out+channel_out
        
        return out
    
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
    

class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=256, squeeze=2):
        super().__init__()
        self.squeeze=squeeze
        self.ch_wv=ConvBlock(channel,channel//self.squeeze,kernel_size=(1,1))
        self.ch_wq=ConvBlock(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=ConvBlock(channel//self.squeeze,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=ConvBlock(channel,channel//self.squeeze,kernel_size=(1,1))
        self.sp_wq=ConvBlock(channel,channel//self.squeeze,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x1):
        b, c, h, w = x1.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x1) #bs,c//2,h,w
        channel_wv=channel_wv.reshape(b,c//self.squeeze,-1) #bs,c//2,h*w
        
        channel_wq=self.ch_wq(x1) #bs,1,h,w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x1

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x1) #bs,c//2,h,w
        spatial_wv=spatial_wv.reshape(b,c//self.squeeze,-1) #bs,c//2,h*w
        
        spatial_wq=self.sp_wq(x1) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//self.squeeze) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x1
        
        
        out=spatial_out+channel_out
        
        return out
    

    




class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
           ConvBlock(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            ConvBlock(channel//reduction,channel,1,bias=False)
        )
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=max_out+avg_out
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=ConvBlock(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=256,reduction=16,kernel_size=7):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)
        self.sigmoid=nn.Sigmoid()
        # self.conv = nn.Sequential(nn.Dropout2d(0.1, False), ConvBlock(channel, channel, 1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ConvBlock):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x1):
        b, c, _, _ = x1.size()
        residual=x1
        
        ca_x1=self.ca(x1)
        out1=x1*self.sigmoid(ca_x1)
        sa_x1=self.sa(out1)
        out1=out1*self.sigmoid(sa_x1)
    
        return out1+residual 


class EncoderRegNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.model = self._load_model()
        self.layers = self._get_layers()
        self.last_layer_channels = 720

    def _load_model(self) -> Any:
        model = regnet_x_8gf(pretrained=self.pretrained)
        return model

    def _get_layers(self) -> List[Any]:
        layers = [
            self.model.stem,
            self.model.trunk_output.block1,
            self.model.trunk_output.block2,
            self.model.trunk_output.block3
        ]
        return layers

    def forward(self, x: Any) -> List[Any]:
        encoder_maps = []
        for layer in self.layers:
            x = layer(x)
            encoder_maps.append(x)
        return encoder_maps


class EncoderSpikingResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_layer_channels = 1024
        self.model = self._load_model()
        self.layers = self._get_layers()
        

    def _load_model(self) -> Any:
        model = spiking_resnet.spiking_nfresnet50(neuron=neuron.SLTTNeuron)
        return model

    def _get_layers(self) -> List[Any]:
        layers = [
            self.model.conv1,
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


class SelfAttention(nn.Module):

  def __init__(self, curr_dim, input_dim):
    super(SelfAttention, self).__init__()
    
    self.input_proj = nn.Conv2d(curr_dim, input_dim , kernel_size=1)
    
    self.input_dim = input_dim
    self.query = nn.Linear(input_dim, input_dim) # [batch_size, seq_length, input_dim]
    self.key = nn.Linear(input_dim, input_dim) # [batch_size, seq_length, input_dim]
    self.value = nn.Linear(input_dim, input_dim)
    self.softmax = nn.Softmax(dim=2)
    
    self.out_proj = nn.Conv2d(input_dim, curr_dim, kernel_size=1)
   
  def forward(self, x): # x.shape (batch_size, seq_length, input_dim)
    

    proj_x = self.input_proj(x)
    
    proj_b, proj_c, proj_h, proj_w = proj_x.shape
    flat_x = proj_x.flatten(2).permute(0,2,1)
    
    queries = self.query(flat_x)
    keys = self.key(flat_x)
    values = self.value(flat_x)

    score = torch.bmm(queries.transpose(1, 2), keys)/(self.input_dim**0.5)
    attention = self.softmax(score)
    weighted = torch.bmm(values, attention).permute(0,2,1).view(proj_b, proj_c, proj_h, proj_w)
    out = self.out_proj(weighted)
    return out + x



class SpatialSelfAttention(nn.Module):
    
    """
    attention module that outputs self and cross attention
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        
        self.q = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        
        self.v = nn.Conv2d(in_channels,
                            in_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        
        
        self.proj_out_self = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        
        
    def attention_calc(self, q, k):

        # compute attention
        b,c,h,w = q.shape
        q = torch.permute(q, (0,2,3,1)).view(b,h*w,c) #rearrange(q, 'b c h w -> b (h w) c')
        k = k.view(b,c,h*w) # rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.bmm(q,k) # torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (c**(-0.5))
       
        return w_

    def proc_self_attention(self, v, w_, h, reshape):
        
        # attend to values
        w_ = torch.permute(w_, (0,2,1)) #rearrange(w_, 'b i j -> b j i')
        h_ = torch.bmm(v,w_) #torch.einsum('bij,bjk->bik', v, w_)
        h_ = h_.view(reshape[0], reshape[1], reshape[2], reshape[3]) #rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out_self(h_)
        return h_
        
    def forward(self, x):
        
        # spatial self-attention
        h_ = x
        q = self.q(h_) 
        k = self.k(h_)
        w_ = self.attention_calc(q,k)
        v = self.v(h_)
        b,c,h,w = v.shape
        shape = v.shape
        v = v.view(b,c,h*w) #rearrange(v, 'b c h w -> b c (h w)')
        h_ = self.proc_self_attention(v, torch.nn.functional.softmax(w_, dim=2), h, shape)
    
        return x+h_
    
class SpatialSelfCrossAttention(nn.Module):
    
    """
    attention module that outputs self and cross attention
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        
        self.q1 = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        
        self.v = nn.Conv2d(in_channels,
                            in_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        
        self.q2= nn.Conv2d(in_channels,
                            in_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        
        self.proj_out_self = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        
        
        self.proj_out_cross = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        
    def attention_calc(self, q, k):

        # compute attention
        b,c,h,w = q.shape
        q = torch.permute(q, (0,2,3,1)).view(b,h*w,c) #rearrange(q, 'b c h w -> b (h w) c')
        k = k.view(b,c,h*w) # rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.bmm(q,k) # torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (c**(-0.5))
       
        return w_

    def proc_self_attention(self, v, w_, h, reshape):
        
        # attend to values
        w_ = torch.permute(w_, (0,2,1)) #rearrange(w_, 'b i j -> b j i')
        h_ = torch.bmm(v,w_) #torch.einsum('bij,bjk->bik', v, w_)
        h_ = h_.view(reshape[0], reshape[1], reshape[2], reshape[3]) #rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out_self(h_)
        return h_
    
    def proc_cross_attention(self, v, w_, h, reshape):
        
        # attend to values
        w_ = torch.permute(w_, (0,2,1)) #rearrange(w_, 'b i j -> b j i')
        h_ = torch.bmm(v,w_) # torch.einsum('bij,bjk->bik', v, w_)
        h_ = h_.view(reshape[0], reshape[1], reshape[2], reshape[3]) #rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out_cross(h_)
        return h_
        
    def forward(self, x1, x2):
        
        
        
        # spatial self-attention
        h_1 = x1
        q1 = self.q1(h_1) 
        k = self.k(h_1)
        w_1 = self.attention_calc(q1,k)
        v = self.v(h_1)
        b,c,h,w = v.shape
        shape = v.shape
        v = v.view(b,c,h*w) #rearrange(v, 'b c h w -> b c (h w)')
        h_1 = self.proc_self_attention(v, torch.nn.functional.softmax(w_1, dim=2), h, shape)
        
        # spatial cross-attention
        h_2 = x2
        q2 = self.q2(h_2)
        w_2 = self.attention_calc(q2,k)
        h_12 = self.proc_cross_attention(v, torch.nn.functional.softmax(w_1+w_2, dim=2), h, shape)

        
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
        
        # self.cls_dw = Correlation2xConcat(num_channels=outchannels, conv_block=conv_block)
        # self.reg_dw = Correlation2xConcat(num_channels=outchannels, conv_block=conv_block)
        
        self.cls_dw = CorrelationConcat(num_channels=outchannels)
        self.reg_dw = CorrelationConcat(num_channels=outchannels)
        
        # box pred head
        for i in range(4):
            tower.append(ConvBlock(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            tower.append(nn.BatchNorm2d(outchannels))
            tower.append(nn.ReLU())
            # tower.append(FastParallelPolarizedSelfAttention(outchannels))

        # cls tower
        for i in range(towernum):
            cls_tower.append(ConvBlock(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            cls_tower.append(nn.BatchNorm2d(outchannels))
            cls_tower.append(nn.ReLU())
            # cls_tower.append(FastParallelPolarizedSelfAttention(outchannels))

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
        cls_x = self.cls_encode(search)  # [z11, z12, z13]
        # cls_d = self.cls_encode(dynamic)
        
        reg_z = kernel.reshape(kernel.size(0), kernel.size(1), -1)
        reg_x = self.reg_encode(search)  # [x11, x12, x13]
        # reg_d = self.reg_encode(dynamic)
        
        # cls and reg DW
        cls_dw = self.cls_dw(cls_z, cls_x, search_org ) #, cls_d) #, gaussian_val)
        reg_dw = self.reg_dw(reg_z, reg_x, search_org) #, reg_d) #, gaussian_val)
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
        # self.att = MobileViTBlockv3_v2(num_channels, num_channels, 2, 1, patch_size=(2,2))
        self.att = SelfAttention(num_channels, num_channels) # ParallelPolarizedSelfAttention(num_channels)
        # self.att = FastParallelPolarizedSelfAttention(num_channels, 1)
        
    def forward(self, z, x, d):
        
        b, c, w, h = x.size()
        s = torch.matmul(z.permute(0, 2, 1), x.view(b, c, -1)).view(b, -1, w, h)
        s = torch.cat([s, d], dim=1)
        s = self.enc(s)
        s = self.att(s)
        return s   

    
    
class Correlation(nn.Module):
    """
    Correlation module
    """

    def __init__(self, num_channels: int, num_corr_channels: int = 64,):
        super().__init__()
        
        in_size = num_corr_channels
        self.enc = nn.Sequential(
            ConvBlock(in_size, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        
        
    def forward(self, z, x):
        b, c, w, h = x.size()
        s = torch.matmul(z.permute(0, 2, 1), x.view(b, c, -1)).view(b, -1, w, h)
        s = self.enc(s)
        return s
    

# class Correlation2xConcat(nn.Module):
#     """
#     Correlation module
#     """

#     def __init__(self, num_channels: int, num_corr_channels: int = 128, conv_block: str = "regular", gaussian_map=False):
#         super().__init__()
        
#         self.gaussian_map = gaussian_map
#         in_size = num_channels + num_corr_channels 
#         self.att = LinearSelfAttention() SelfAttention(in_size, num_channels) #FastParallelPolarizedSelfAttention(in_size)
#         self.enc = nn.Sequential(
#             ConvBlock(in_size, num_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(num_channels),
#             nn.ReLU(inplace=True),
#         )
        
        
#     def forward(self, z, x, d):
#         b, c, w, h = x.size()
#         s1 = torch.matmul(z.permute(0, 2, 1), x.view(b, c, -1)).view(b, -1, w, h)
#         s2 = torch.matmul(z.permute(0, 2, 1), d.view(b, c, -1)).view(b, -1, w, h)
#         s = torch.cat([s1, s2, d], dim=1) 
#         s = self.att(s)
#         s = self.enc(s)
#         return s
    