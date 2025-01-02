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
from model.NAT import nat_mini, nat_base
from model.VAN import van_b2



class BasicBlockSem(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(BasicBlockSem, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.ca = eca_layer()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        # Channel Attention Module
        out = self.ca(out) #* out
        out = self.relu(out)

        return out

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

class AngBranch(nn.Module):

    def __init__(self):
        super(AngBranch, self).__init__()

        self.in_block_sem_1 = BasicBlockSem(32, 64, kernel_size=3, stride=2, padding=1)
        self.in_block_sem_2 = BasicBlockSem(64, 128, kernel_size=3, stride=2, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        y1 = self.in_block_sem_1(x)
        y2 = self.in_block_sem_2(y1)
        return y1, y2
    
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
    def __init__(self, image_size, in_channels, patch_size, emb_size, reduction_ratio, swin_window_size, num_heads, swin_blocks,
                num_stb):
        super(IntegratedModelV2, self).__init__()

        self.SFE = nn.Conv2d(3, 3, kernel_size=3, stride=1, dilation=7, padding=7, bias=False)
        self.AFE = nn.Conv2d(3, 32, kernel_size=7, stride=7, padding=0, bias=False)
        self.AngBranch = AngBranch()


        self.nat = van_b2(pretrained=True)  
        #self.cam1 = eca_layer()#(in_planes=256, ratio=20)
        #self.cam2 = eca_layer()#(in_planes=512, ratio=20)


        self.rerange_layer = Rearrange('b c h w -> b (h w) c')
        self.avg_pool = nn.AdaptiveAvgPool2d(224 // 32)

        # Adaptive head
        embed_dim = 1088
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

        

    def forward(self, x_sai, x_mli):
        
        x_ang = self.AFE(x_mli)
        #print('ANG', x_ang.shape)

        a1, a2 = self.AngBranch(x_ang)
        a1 = self.avg_pool(a1)
        a2 = self.avg_pool(a2)


        x_spa = self.SFE(x_mli)
        layer1_s, layer2_s, layer3_s, layer4_s = self.nat(x_spa)    # (b,64,56,56); (b,128,28,28); (b,320,14,14); (b,512,7,7)
        s1 = self.avg_pool(layer1_s)
        s2 = self.avg_pool(layer2_s)
        s3 = self.avg_pool(layer3_s)
        s4 = self.avg_pool(layer4_s)
        #print('SPA', x_spa.shape)
        

        feats = torch.cat((s1, s2, s3, s4, a1, a2), dim=1)
        feats = self.rerange_layer(feats)  # (b, c, h, w) -> (b, h*w, c)
        #assert feats.shape[-1] == 1216 and len(feats.shape) == 3, 'Unexpected stacked features: {}'.format(feats.shape)

        scores = self.head_score(feats)
        weights = self.head_weight(feats)
        q = torch.sum(scores * weights, dim=1) / torch.sum(weights, dim=1)

        return q
