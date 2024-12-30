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
from torchvision.models.video.swin_transformer import SwinTransformer3d

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

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channel: int = 3,
        embed_dim: int = 128,
        kernel_size: int = 7,
        stride: int = 4,
        padding: int = 3,
    ):
        """
        in_channels: number of the channels in the input volume
        embed_dim: embedding dimmesion of the patch
        """
        super().__init__()
        self.patch_embeddings = nn.Conv3d(
            in_channel,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # standard embedding patch
        patches = self.patch_embeddings(x)
        #patches = patches.flatten(2).transpose(1, 2)
        #patches = self.norm(patches)
        return patches


class IntegratedModelV2(nn.Module):
    def __init__(self):
        super(IntegratedModelV2, self).__init__()
        
        self.van = van_b0(pretrained=True)  
        #self.embed = PatchEmbedding()       
        self.swin = SwinTransformer3d(
                    patch_size=(2, 4, 4),  # Patch size (time, height, width)
                    embed_dim=96,          # Embedding dimension
                    depths=[2, 2, 6, 2],   # Number of layers in each stage
                    num_heads=[3, 6, 12, 24],  # Number of attention heads per stage
                    window_size=[8, 7, 7],     # Window size for attention (time, height, width)
                    mlp_ratio=4.0,             # Ratio of hidden size to embedding size in MLP
                    #num_classes=None,           # Number of output classes (set to `None` if not for classification)
                    dropout=0.0,               # Dropout rate
                    attention_dropout=0.0,     # Dropout rate for attention weights
                    stochastic_depth_prob=0.1  # Stochastic depth rate for regularization
                )

        embed_dim = 400
        # Adaptive head
        self.head_score = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            #nn.ReLU()
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
        x = x.unfold(2, 224, 224).unfold(3, 224, 224).permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, 3, 224, 224)
        print(x.shape)   
        x = x.permute(0, 2, 3, 4, 1)
        print(x.shape)   
        x = self.swin(x)
        print(x.shape)   

        
        #x = x.reshape(batch_size * 6, 3, 512, 512)        
        #s1, s2, s3, s4 = self.van(x)
        #s1 = s1.reshape(batch_size, 6, 32, 128, 128).permute(0, 2, 1, 3, 4)
        #s2 = s2.reshape(batch_size, 6, 64, 64, 64).permute(0, 2, 1, 3, 4)
        #s3 = s3.reshape(batch_size, 6, 160, 32, 32).permute(0, 2, 1, 3, 4)
        #s4 = s4.reshape(batch_size, 6, 256, 16, 16).permute(0, 2, 1, 3, 4)

        

        #x = x.view(batch_size, 6 * 256, 16, 16)
        #x = rearrange(x, 'b c h w-> b (h w) c')  


        #print(x.shape)

        scores = self.head_score(x)
        
        return scores
    