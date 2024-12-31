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



class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
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
        

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []



class IntegratedModelV2(nn.Module):
    def __init__(self):
        super(IntegratedModelV2, self).__init__()
        
        #self.van = van_b0(pretrained=True)  
        self.deit = create_model('deit_tiny_patch16_224', pretrained=True)

        self.save_output = SaveOutput()

        # Freeze all layers
        for param in self.deit.parameters():
            param.requires_grad = False

        hook_handles = []
        for layer in self.deit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        patch_size=(2, 4, 4)
        embed_dim=96
      
        self.swin = nn.ModuleList([
             SwinTransformer3d(
                    in_channels=30,
                    patch_size=patch_size,  # Patch size (time, height, width)
                    embed_dim=embed_dim,          # Embedding dimension
                    depths=[1, 1],   # Number of layers in each stage
                    num_heads=[2, 2],  # Number of attention heads per stage
                    window_size=[2, 4,4],     # Window size for attention (time, height, width)
                    mlp_ratio=4.0,             # Ratio of hidden size to embedding size in MLP
                    num_classes=embed_dim,           # Number of output classes (set to `None` if not for classification)
                    dropout=0.2,               # Dropout rate
                    attention_dropout=0.2,     # Dropout rate for attention weights
                    #stochastic_depth_prob=0.1  # Stochastic depth rate for regularization
                )
                for _ in range(1)
        ])
    

        embed = embed_dim *1
        # Adaptive head
        self.head_score = nn.Sequential(
            nn.Linear(embed, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            #nn.ReLU()
        )

    def forward(self, x):
        
        batch_size = x.shape[0]
        x = x.unfold(2, 224, 224).unfold(3, 224, 224).permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, 3, 224, 224)

        x = x.reshape(batch_size * 30, 3, 224, 224)     
        x = self.deit(x)   
        x = self.save_output.outputs[11][:, 1:]
        self.save_output.outputs.clear()

        print(x.shape)
        x = x.reshape(batch_size, 30, 14, 14, 192).permute(0,1,4,2,3)
        for swin_block in self.swin:
            x = swin_block(x)
         
        print(x.shape)

        scores = self.head_score(x)
        print(scores.shape)
        
        return scores
    