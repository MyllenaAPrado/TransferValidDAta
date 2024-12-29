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




class LFIQA(nn.Module):
    def __init__(self):
        super(LFIQA, self).__init__()
        
        self.van = van_b0(pretrained=True) 

        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*56*56, 512),
            nn.ELU(),
            nn.Linear(512, 64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        
        batch_size = x.shape[0]
        x = x.unfold(2, 224, 224).unfold(3, 224, 112).permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, 3, 224, 224)
        x = x.view(batch_size * 15*3, 3, 224, 224)        
        x,_, _ , _ = self.van(x)
        print(x.shape)
        x = x.view(batch_size, 15*3, 32, 56, 56)

        print(x.shape)
        x = x.view(batch_size, 45, 56*56*32)
        print(x.shape)
        print(x.shape)

        x = self.regression(x)     

        return x
    