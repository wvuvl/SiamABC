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

# class SpatialSelfCrossAttention(nn.Module):
    
#     """
#     attention module that outputs self and cross attention
#     """
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels
        
        
#         self.attention_channel = Attention(in_channels=in_channels)
#         self.v_channel = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
#         self.proj_out_self_channel = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
#         self.proj_out_cross_channel = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
        
#         self.attention_spatial = Attention(in_channels=in_channels)
#         self.v_spatial = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
#         self.proj_out_self_spatial = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
#         self.proj_out_cross_spatial = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
        

#     def proc_self_attention_channel(self, v, w_, h):
        
#         # attend to values
#         w_ = rearrange(w_, 'b i j -> b j i')
#         h_ = torch.einsum('bij,bjk->bik', v, w_)
#         h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
#         h_ = self.proj_out_self_channel(h_)
#         return h_
    
#     def proc_cross_attention_channel(self, v, w_, h):
        
#         # attend to values
#         w_ = rearrange(w_, 'b i j -> b j i')
#         h_ = torch.einsum('bij,bjk->bik', v, w_)
#         h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
#         h_ = self.proj_out_cross_channel(h_)
#         return h_
    
#     def get_channel_attention(self,x1,x2):
#         # channel self-attention
#         h_1 = x1
#         w_1 = self.attention_channel(h_1)
#         v = self.v_channel(h_1)
#         b,c,h,w = v.shape
#         v = rearrange(v, 'b c h w -> b c (h w)')
#         h_1 = self.proc_self_attention_channel(v, torch.nn.functional.softmax(w_1, dim=1), h)
        
#         # channel cross-attention
#         h_2 = x2
#         w_2 = self.attention_channel(h_2)
#         h_12 = self.proc_cross_attention_channel(v, torch.nn.functional.softmax(w_1+w_2, dim=1), h)
#         return h_1, h_12
    
    
    
#     def proc_self_attention_spatial(self, v, w_, h):
        
#         # attend to values
#         w_ = rearrange(w_, 'b i j -> b j i')
#         h_ = torch.einsum('bij,bjk->bik', v, w_)
#         h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
#         h_ = self.proj_out_self_spatial(h_)
#         return h_
    
#     def proc_cross_attention_spatial(self, v, w_, h):
        
#         # attend to values
#         w_ = rearrange(w_, 'b i j -> b j i')
#         h_ = torch.einsum('bij,bjk->bik', v, w_)
#         h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
#         h_ = self.proj_out_cross_spatial(h_)
#         return h_
    
#     def get_spatial_attention(self,x1,x2):
#         # spatial self-attention
#         h_1 = x1
#         w_1 = self.attention_spatial(h_1)
#         v = self.v_spatial(h_1)
#         b,c,h,w = v.shape
#         v = rearrange(v, 'b c h w -> b c (h w)')
#         h_1 = self.proc_self_attention_spatial(v, torch.nn.functional.softmax(w_1, dim=2), h)
        
#         # spatial cross-attention
#         h_2 = x2
#         w_2 = self.attention_spatial(h_2)
#         h_12 = self.proc_cross_attention_spatial(v, torch.nn.functional.softmax(w_1+w_2, dim=2), h)
#         return h_1, h_12
        
#     def forward(self, x1, x2):
#         self_att_spatial_weight, cross_att_spatial_weight = self.get_spatial_attention(x1,x2)
#         # self_att_spatial, cross_att_saptial = x1+self_att_spatial_weight, x1+cross_att_spatial_weight
        
#         self_att_channel_weight, cross_att_channel_weight = self.get_channel_attention(x1,x2)
#         # self_att_channel, cross_att_channel = x1+self_att_channel_weight, x1+cross_att_channel_weight
        
        
#         return x1+self_att_spatial_weight+self_att_channel_weight, x1+cross_att_spatial_weight+cross_att_channel_weight
# class ParallelPolarizedCrossAttention(nn.Module):

#     def __init__(self, channel=256):
#         super().__init__()
#         self.ch_wv=ConvBlock(channel,channel//2,kernel_size=(1,1))
#         self.ch_wq=ConvBlock(channel,1,kernel_size=(1,1))
#         self.softmax_channel=nn.Softmax(1)
#         self.softmax_spatial=nn.Softmax(-1)
#         self.ch_wz=ConvBlock(channel//2,channel,kernel_size=(1,1))
#         self.ln=nn.LayerNorm(channel)
#         self.sigmoid=nn.Sigmoid()
#         self.sp_wv=ConvBlock(channel,channel//2,kernel_size=(1,1))
#         self.sp_wq=ConvBlock(channel,channel//2,kernel_size=(1,1))
#         self.agp=nn.AdaptiveAvgPool2d((1,1))


#     def run_channel_attention(self, x1):
        
#         b, c, h, w = x1.size()
        
#         #Channel-only Self-Attention
#         channel_wv1=self.ch_wv(x1) #bs,c//2,h,w
#         channel_wv1=channel_wv1.reshape(b,c//2,-1) #bs,c//2,h*w
        
#         channel_wq1=self.ch_wq(x1) #bs,1,h,w
#         channel_wq1=channel_wq1.reshape(b,-1,1) #bs,h*w,1
#         channel_wq1=self.softmax_channel(channel_wq1)
                
#         return channel_wv1, channel_wq1
    
#     def run_spatial_attention(self,x1):
        
#         b, c, h, w = x1.size()
        
#         #Spatial-only Self-Attention
#         spatial_wv1=self.sp_wv(x1) #bs,c//2,h,w
#         spatial_wv1=spatial_wv1.reshape(b,c//2,-1) #bs,c//2,h*w
        
#         spatial_wq1=self.sp_wq(x1) #bs,c//2,h,w
#         spatial_wq1=self.agp(spatial_wq1) #bs,c//2,1,1
#         spatial_wq1=spatial_wq1.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
#         spatial_wq1=self.softmax_spatial(spatial_wq1)
        
#         return spatial_wq1, spatial_wv1
        
#     def forward(self, x1, x2):
#         b, c, h, w = x1.size()
#         channel_wv1, channel_wq1 = self.run_channel_attention(x1)
#         channel_wz1=torch.matmul(channel_wv1,channel_wq1).unsqueeze(-1) #bs,c//2,1,1
#         channel_weight1=self.sigmoid(self.ln(self.ch_wz(channel_wz1).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
#         channel_out1=channel_weight1*x1
        
#         spatial_wq1, spatial_wv1 = self.run_spatial_attention(x1)
#         spatial_wz1=torch.matmul(spatial_wq1,spatial_wv1) #bs,1,h*w
#         spatial_weight1=self.sigmoid(spatial_wz1.reshape(b,1,h,w)) #bs,1,h,w
#         spatial_out1=spatial_weight1*x1
        
#         self_att=spatial_out1+channel_out1
        
        
#         channel_wv2, channel_wq2 = self.run_channel_attention(x2)
#         channel_wz2=torch.matmul(channel_wv1,channel_wq2).unsqueeze(-1) #bs,c//2,1,1
#         channel_weight2=self.sigmoid(self.ln(self.ch_wz(channel_wz2).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
#         channel_out2=channel_weight2*x1
        
#         spatial_wq2, spatial_wv2 = self.run_spatial_attention(x2)
#         spatial_wz2=torch.matmul(spatial_wq2,spatial_wv1) #bs,1,h*w
#         spatial_weight2=self.sigmoid(spatial_wz2.reshape(b,1,h,w)) #bs,1,h,w
#         spatial_out2=spatial_weight2*x1
        
#         cross_att=spatial_out2+channel_out2
        
        
#         return self_att, cross_att
    
    

class ParallelPolarizedSelfCrossAttention(nn.Module):

    def __init__(self, channel=256):
        super().__init__()
        self.ch_wv=ConvBlock(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=ConvBlock(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=ConvBlock(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=ConvBlock(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=ConvBlock(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))


    def run_channel_attention(self, x1):
        
        b, c, h, w = x1.size()
        
        #Channel-only Self-Attention
        channel_wv1=self.ch_wv(x1) #bs,c//2,h,w
        channel_wv1=channel_wv1.reshape(b,c//2,-1) #bs,c//2,h*w
        
        channel_wq1=self.ch_wq(x1) #bs,1,h,w
        channel_wq1=channel_wq1.reshape(b,-1,1) #bs,h*w,1
        channel_wq1=self.softmax_channel(channel_wq1)
        
        channel_wz1=torch.matmul(channel_wv1,channel_wq1).unsqueeze(-1) #bs,c//2,1,1
        channel_weight1=self.sigmoid(self.ln(self.ch_wz(channel_wz1).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        
        return channel_weight1
    
    def run_spatial_attention(self,x1):
        
        b, c, h, w = x1.size()
        
        #Spatial-only Self-Attention
        spatial_wv1=self.sp_wv(x1) #bs,c//2,h,w
        spatial_wv1=spatial_wv1.reshape(b,c//2,-1) #bs,c//2,h*w
        
        spatial_wq1=self.sp_wq(x1) #bs,c//2,h,w
        spatial_wq1=self.agp(spatial_wq1) #bs,c//2,1,1
        spatial_wq1=spatial_wq1.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq1=self.softmax_spatial(spatial_wq1)
        
        spatial_wz1=torch.matmul(spatial_wq1,spatial_wv1) #bs,1,h*w
        spatial_weight1=self.sigmoid(spatial_wz1.reshape(b,1,h,w)) #bs,1,h,w
        
        return spatial_weight1
        
    def forward(self, x1, x2):
        channel_out1=self.run_channel_attention(x1)*x1
        spatial_out1=self.run_spatial_attention(x1)*x1
        self_att=spatial_out1+channel_out1
        
        channel_out2=self.run_channel_attention(x2)*x1
        spatial_out2=self.run_spatial_attention(x2)*x1
        cross_att=spatial_out2+channel_out2
        
        
        return self_att, cross_att


class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=256):
        super().__init__()
        self.ch_wv=ConvBlock(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=ConvBlock(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=ConvBlock(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=ConvBlock(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=ConvBlock(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x1):
        b, c, h, w = x1.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x1) #bs,c//2,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        
        channel_wq=self.ch_wq(x1) #bs,1,h,w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x1

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x1) #bs,c//2,h,w
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        
        spatial_wq=self.sp_wq(x1) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x1
        
        
        out=spatial_out+channel_out
        
        return out
    

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = ConvBlock(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = ConvBlock(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = ConvBlock(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
    
class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class DANetAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(ConvBlock(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(ConvBlock(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(ConvBlock(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(ConvBlock(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), ConvBlock(inter_channels, out_channels, 1))

    def forward(self, x1):
        feat_x1_1 = self.conv5a(x1)
        sa_feat_x1_1 = self.sa(feat_x1_1)
        sa_conv_x1_1 = self.conv51(sa_feat_x1_1)

        featx1_2 = self.conv5c(x1)
        sc_x1_feat = self.sc(featx1_2)
        sc_x1_conv = self.conv52(sc_x1_feat)

        feat_sum = sa_conv_x1_1+sc_x1_conv
        
        sasc_output = self.conv6(feat_sum)

        return sasc_output



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
        
        # ca_x2=self.ca(x2)
        # out2=x2*self.sigmoid(ca_x2)
        # sa_x2=self.sa(out2)
        # out2=out2*self.sigmoid(sa_x2)

        # cross_out=self.sigmoid(self.conv(out1+out2))
        return out1+residual #, cross_out+residual


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




class ChanAttention(nn.Module):

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
    
    def proc_self_attention(self, v, w_, h):
        
        # attend to values
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b (h w) c -> b c h w', h=h)
        h_ = self.proj_out_self(h_)
        return h_
    
    def forward(self, x):
        h_ = x
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b (h w) c')
        v = rearrange(v, 'b c h w -> b (h w) c')
        
        w_ = torch.einsum('bij,bjk->bik', q, k)
        w_ = w_ * (int(c)**(-0.5))
        
        h_ = self.proc_self_attention(v, torch.nn.functional.softmax(w_, dim=2), h)
        
        return x+h_
    
    
class Attention(nn.Module):
    """
    modified from
    src: https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/attention.py#L126
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
        self.attention = Attention(in_channels=in_channels)
        # self.cbam_attention = DANetAttention(in_channels=in_channels, out_channels=in_channels) #CBAMBlock(channel=in_channels)
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
        
        
        self.proj_out_cross = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        

    def proc_self_attention(self, v, w_, h):
        
        # attend to values
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out_self(h_)
        return h_
    
    def proc_cross_attention(self, v, w_, h):
        
        # attend to values
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out_cross(h_)
        return h_
        
    def forward(self, x1, x2):
        
        # spatial self-attention
        h_1 = x1
        w_1 = self.attention(h_1)
        v = self.v(h_1)
        b,c,h,w = v.shape
        v = rearrange(v, 'b c h w -> b c (h w)')
        h_1 = self.proc_self_attention(v, torch.nn.functional.softmax(w_1, dim=2), h)
        
        # spatial cross-attention
        h_2 = x2
        w_2 = self.attention(h_2)
        h_12 = self.proc_cross_attention(v, torch.nn.functional.softmax(w_1+w_2, dim=2), h)

        
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
        
        self.cls_dw = Correlation2xConcat(num_channels=outchannels, conv_block=conv_block)
        self.reg_dw = Correlation2xConcat(num_channels=outchannels, conv_block=conv_block)
        
        # self.cls_dw = CorrelationConcat(num_channels=outchannels, conv_block=conv_block)
        # self.reg_dw = CorrelationConcat(num_channels=outchannels, conv_block=conv_block)
        
        # box pred head
        for i in range(towernum):
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

    def forward(self, search, dynamic, kernel, gaussian_val=None):
        # encode first
        cls_z = kernel.reshape(kernel.size(0), kernel.size(1), -1)
        cls_x = self.cls_encode(search)  # [z11, z12, z13]
        cls_d = self.cls_encode(dynamic)
        
        reg_z = kernel.reshape(kernel.size(0), kernel.size(1), -1)
        reg_x = self.reg_encode(search)  # [x11, x12, x13]
        reg_d = self.reg_encode(dynamic)
        
        # cls and reg DW
        cls_dw = self.cls_dw(cls_z, cls_x, cls_d) #, gaussian_val)
        reg_dw = self.reg_dw(reg_z, reg_x, reg_d) #, gaussian_val)
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

    def __init__(self, num_channels: int, num_corr_channels: int = 64, conv_block: str = "regular", gaussian_map=False):
        super().__init__()
        
        self.gaussian_map = gaussian_map
        in_size = num_channels + num_corr_channels 
        if self.gaussian_map:  
            in_size = in_size + 2 # 2 chan gaussian map, 1 for t-2 and 2 for t-1 
            self.weight = nn.Parameter(torch.empty(1, 2, 16, 16)) #gaussian map channels
            trunc_normal_(self.weight, std=.02)
            
        self.enc = nn.Sequential(
            ConvBlock(in_size, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        
        
    def forward(self, z, x, d, g=None):
        
        b, c, w, h = x.size()
        s = torch.matmul(z.permute(0, 2, 1), x.view(b, c, -1)).view(b, -1, w, h)
        s = torch.cat([s, d], dim=1) if g==None else torch.cat([s, d, g*self.weight], dim=1) # applying a broadcast weight factor to the gaussian_map parameter g (b, 2, 16, 16)
        s = self.enc(s)
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
    

class Correlation2xConcat(nn.Module):
    """
    Correlation module
    """

    def __init__(self, num_channels: int, num_corr_channels: int = 128, conv_block: str = "regular", gaussian_map=False):
        super().__init__()
        
        self.gaussian_map = gaussian_map
        in_size = num_channels + num_corr_channels 
        # self.att = ParallelPolarizedSelfAttention(in_size)
        self.enc = nn.Sequential(
            ConvBlock(in_size, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        
        
    def forward(self, z, x, d):
        b, c, w, h = x.size()
        s1 = torch.matmul(z.permute(0, 2, 1), x.view(b, c, -1)).view(b, -1, w, h)
        s2 = torch.matmul(z.permute(0, 2, 1), d.view(b, c, -1)).view(b, -1, w, h)
        s = torch.cat([s1, s2, d], dim=1) 
        # s = self.att(s)
        s = self.enc(s)
        return s
    