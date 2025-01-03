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
#from model.NAT import nat_mini, nat_base
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


class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []


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
        
        #self.nat = nat_base(pretrained=True)  
        self.vit =  timm.create_model('vit_base_patch8_224', pretrained=True)

        self.save_output = SaveOutput()

        # Freeze all layers
        for param in self.vit.parameters():
            param.requires_grad = False

        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.eca = eca_layer()
        self.eca2 = eca_layer()

        self.rerange_layer = Rearrange('b c h w -> b (h w) c')
        self.avg_pool = nn.AdaptiveAvgPool2d(224 // 32)

        self.conv = nn.Conv2d(in_channels=26*3, out_channels=256, kernel_size=6, stride = 6)    

        embed_dim = 282
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

    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unfold(2, 224, 224).unfold(3, 224, 224).permute(0, 2, 3, 1, 4, 5).reshape(batch_size,-1, 3, 224, 224)

        x_nat = x.reshape(batch_size*26,3, 224, 224)
        #x_nat = x[:, 13, :, :, :].reshape(batch_size,3, 224, 224) 
        #_, s2, _, s4 = self.nat(x_nat) 
        #x1 = s2.permute(0,3, 1, 2)
        #x2 = s4.permute(0,3, 1, 2)
        #x1 = self.avg_pool(x1)
        #x2 = self.avg_pool(x2)
        x_nat = self.vit(x_nat)  # Features shape: (batch_size * 16, embed_dim)
        x_nat = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()
        print(x_nat.shape)
        x_nat = x.reshape(batch_size, 26, 196, 768)
        x_nat = self.avg_pool(x_nat)
        print(x_nat.shape)

        x_eca = x.reshape(batch_size, 26*3, 224, 224)   
        x_eca = self.eca (x_eca)
        x_eca = self.conv (x_eca)
        x_eca = self.eca2 (x_eca)
        x_eca = self.avg_pool(x_eca)
       
        feats = torch.cat((x_eca, x_nat), dim=1)
        feats = self.rerange_layer(feats)  # (b, c, h, w) -> (b, h*w, c)
        scores = self.head_score(feats)
        weights = self.head_weight(feats)
        q = torch.sum(scores * weights, dim=1) / torch.sum(weights, dim=1)

        return q
    
