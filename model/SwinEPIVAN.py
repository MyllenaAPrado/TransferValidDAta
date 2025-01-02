import torch
import torch.nn as nn
import timm
import numpy as np
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data, Batch
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from model.VAN import van_b0, van_b1
from einops import rearrange
from model.Swin3D import SwinTransformer3d
from timm import create_model
from torch.nn.parameter import Parameter
from timm.models.swin_transformer import SwinTransformerBlock, PatchMerging
from model.NAT import nat_mini, nat_base
from einops.layers.torch import Rearrange

class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention3D, self).__init__()
        # Adaptive pooling for 3D inputs
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # Fully connected layers replaced with 3D convolution
        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, kernel_size=1, bias=False)

        # Non-linearity and activation
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply average pooling and max pooling
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        
        # Combine outputs
        out = avg_out + max_out
        return self.sigmoid(out)




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


class IntegratedModelV2(nn.Module):
    def __init__(self):
        super(IntegratedModelV2, self).__init__()
        
        
        self.nat = nat_base(pretrained=True)  

        self.cam1 = ChannelAttention3D(in_planes=512, ratio=20)
        self.cam2 = ChannelAttention3D(in_planes=1024, ratio=20)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.rerange_layer = Rearrange('b c h w d -> b (h w d) c')
        self.avg_pool = nn.AdaptiveAvgPool3d(224 // 32)

        self.conv1 = nn.Conv2d(in_channels=512*6, out_channels=256, kernel_size=2, stride = 2)    
        self.conv2 = nn.Conv2d(in_channels=1024*6, out_channels=256, kernel_size=1)    


        embed_dim = 1536
        # Adaptive head
        self.head_score = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.ReLU()
        )
        self.head_weight = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        batch_size = x.shape[0]
        x = x.unfold(2, 512, 512).unfold(3, 512, 512).permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, 3, 512, 512)
        
        x = x.reshape(batch_size*6, 3, 512, 512)     

        _, s2, s3, s4 = self.nat(x)    

        print(s2.shape)
        print(s3.shape)
        print(s4.shape)

        x1 = s2.reshape(batch_size, 6, 32, 32, 512).permute(0,4, 1, 2, 3)
        x2 = s4.reshape(batch_size, 6, 16, 16, 1024).permute(0,4, 1, 2, 3)

        x1 = self.cam1(x1) * x1
        x2 = self.cam2(x2) * x2
        print(x1.shape)
        print(x2.shape)
        x1 = self.avg_pool(x1)
        x2 = self.avg_pool(x2)
        print(x1.shape)
        print(x2.shape)
        #x1 = x1.reshape(batch_size, 6*512, 32, 32)
        #x2 = x2.reshape(batch_size, 6*1024, 16, 16)
        #x1 = self.conv1(x1)
        #x2 = self.conv2(x2)

        feats = torch.cat((x1,x2), dim=1)
        feats = self.rerange_layer(feats)  # (b, c, h, w) -> (b, h*w, c)
        scores = self.head_score(feats)
        weights = self.head_weight(feats)
        q = torch.sum(scores * weights, dim=1) / torch.sum(weights, dim=1)

        return q
    
