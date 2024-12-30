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


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute average and max pooling along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        # Concatenate along the channel dimension
        concat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        # Pass through convolution and sigmoid
        attention = self.sigmoid(self.conv(concat))  # [B, 1, H, W]
        # Refine the input feature map
        return x * attention


class ECA(nn.Module):
    def __init__(self, channels, gamma=2, beta=1):
        super(ECA, self).__init__()
        # Determine kernel size based on the number of channels
        kernel_size = int(abs((torch.log2(torch.tensor(channels).float()) + beta) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1  # Ensure odd kernel size
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling
        y = x.mean(dim=(-1, -2), keepdim=True)  # [B, C, 1, 1]
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y


class IntegratedModelV2(nn.Module):
    def __init__(self):
        super(IntegratedModelV2, self).__init__()
        
        self.van = van_b0(pretrained=True) 
        self.avg_pool = nn.AdaptiveAvgPool2d(224 // 32)


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
        print(x.shape)   
        
        x = x.reshape(batch_size * 6, 3, 512, 512)        
        _,_, _ , x = self.van(x)
        print(x.shape)
        x = x.view(batch_size, 6 * 256, 16, 16)
        x = rearrange(x, 'b c h w-> b (h w) c')  


        print(x.shape)

        scores = self.head_score(x)
        weights = self.head_weight(x)
        q = torch.sum(scores * weights, dim=1) / torch.sum(weights, dim=1)

        return q
    