import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock, PatchMerging
from einops import rearrange
import torch.nn as nn
import torchvision.models as models

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg
from model.VAN import van_b0, van_b1, van_b2, van_b3  # Adjust import based on where VAN is defined
import math
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm import create_model
from timm.models.vision_transformer import Block
from model.NAT import nat_mini



class ECA3DLayer(nn.Module):
    """Constructs a 3D ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel=3, k_size=3):
        super(ECA3DLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 3D global average pooling
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global spatial information descriptor
        y = self.avg_pool(x)  # Output shape: (B, C, 1, 1, 1)

        # Apply 1D convolution across the channel dimension
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)

        # Apply sigmoid and scale the input
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.relu1 = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class IntegratedModelV2(nn.Module):
    def __init__(self, image_size, in_channels, patch_size, emb_size, reduction_ratio, swin_window_size, num_heads, swin_blocks,
                num_stb):
        super(IntegratedModelV2, self).__init__()

        self.conv_down = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=3
        )

        self.nat = nat_mini(pretrained=True)  
        self.cam1 = eca_layer()#(in_planes=256, ratio=20)
        self.cam2 = eca_layer()#(in_planes=512, ratio=20)


        self.avg_pool = nn.AdaptiveAvgPool2d(224 // 32)
        self.rerange_layer = Rearrange('b c h w -> b (h w) c')

        embed_dim = 768
        # Adaptive head
        self.head_score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim//2, 1),
            nn.ReLU()
        )
        self.head_weight = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

        

    def forward(self, x_sai, x_mli):
        #print(x_mli.shape)
        x_mli = self.conv_down(x_mli)
        #print(x_mli.shape)

        s1, s2, s3, s4 = self.nat(x_mli)    

        #print(s1.shape)
        #print(s2.shape)
        #print(s3.shape)
        #print(s4.shape)
        s2 = rearrange(s2, 'b h w c-> b c h w')  
        s4 = rearrange(s4, 'b h w c-> b c h w')  


        x1 = self.cam1(s2)# * s2
        x2 = self.cam2(s4)# * s4
        #print(x1.shape)
        #print(x2.shape)
        x1 = self.avg_pool(x1)
        x2 = self.avg_pool(x2)
        #print(x1.shape)
        #print(x2.shape)

        feats = torch.cat((x1,x2), dim=1)
        feats = self.rerange_layer(feats)  # (b, c, h, w) -> (b, h*w, c)
        scores = self.head_score(feats)
        weights = self.head_weight(feats)
        q = torch.sum(scores * weights, dim=1) / torch.sum(weights, dim=1)

        return q
