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
from model.NAT import nat_mini


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
        
        
        self.nat = nat_mini(pretrained=True)  

        self.eca1 = eca_layer()
        self.eca2 = eca_layer()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
    

        embed_dim = 143
        # Adaptive head
        self.head_score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim//2, embed_dim),
            nn.ReLU()
        )
        self.head_weight = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        batch_size = x.shape[0]
        x = x.unfold(2, 512, 512).unfold(3, 512, 512).permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, 3, 512, 512)
        print(x.shape)

        _, _, s3, s4 = self.nat(x)    # (b,64,56,56); (b,128,28,28); (b,320,14,14); (b,512,7,7)

        print(s3.shape)
        print(s4.shape)

        x1 = s3.reshape(batch_size, 6, 49, 512)
        x2 = s4.reshape(batch_size, 6, 49, 512)

        x1 = self.eca(x1)
        x2 = self.eca(x2)

        print(x1.shape)
        print(x2.shape)


        
        #x = self.global_pool(x2)  # Reduce spatial dimensions to [batch_size, emb_size, 1, 1]

        feats = torch.cat((x1,x2), dim=1)
        feats = self.rerange_layer(feats)  # (b, c, h, w) -> (b, h*w, c)
        scores = self.head_score(feats)
        weights = self.head_weight(feats)
        q = torch.sum(scores * weights, dim=1) /torch.sum(weights, dim=1)

        return q
    



    '''
    # Apply patch embedding
    x = self.patch_embedding(attended_features)
    #print('Swin input:',x.shape)
    # Rearrange for Swin Transformer
    x = rearrange(x, 'b c h w -> b h w c')

    # Pass through Swin Transformer blocks
    for swin_block in self.swin_blocks:
        x = swin_block(x)

    x = rearrange(x, 'b h w c -> b c h w')
    x = self.eca(x)
    x = rearrange(x, 'b c h w -> b h w c')    

    x = self.patch_merging_1(x)
    for swin_block in self.swin_blocks_1:
        x = swin_block(x)  

    # Global Pooling and Fully Connected Layer
    x = rearrange(x, 'b h w c -> b c h w')  # Restore to [batch_size, emb_size, height, width]
    '''
    