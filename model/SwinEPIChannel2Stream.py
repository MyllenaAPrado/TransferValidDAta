import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock, PatchMerging
from einops import rearrange
import torch.nn as nn
import torchvision.models as models

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg
from model.VAN import van_b0, van_b1  # Adjust import based on where VAN is defined



class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channel: int = 4,
        embed_dim: int = 768,
        kernel_size: int = 7,
        stride: int = 4
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
            stride=stride
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # standard embedding patch
        patches = self.patch_embeddings(x)
        
        patches = patches.permute(0, 2, 1, 3, 4)  # Move depth to come before channels: [B, D, C, H, W]
        B, D, C, H, W = patches.shape
        patches = patches.reshape(B, D * C, H, W)  # Combine depth and channels: [B, D*C, H, W]

        return patches
    



class CAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pooled = self.avg_pool(x).view(b, c)
        max_pooled = self.max_pool(x).view(b, c)
        avg_attention = self.fc2(self.relu(self.fc1(avg_pooled)))
        max_attention = self.fc2(self.relu(self.fc1(max_pooled)))
        attention = self.sigmoid(avg_attention + max_attention).view(b, c, 1, 1)
        return x * attention  # Refine channels by multiplication

class IntegratedModelV2(nn.Module):
    def __init__(self, image_size, in_channels, patch_size, emb_size, reduction_ratio, swin_window_size, num_heads, swin_blocks,
                EAM, num_stb):
        super(IntegratedModelV2, self).__init__()
        self.van = van_b0(pretrained=True)  # Pretrained VAN model (van_b0 or other variant)
        
        # Calculate height and width after patch embedding
        height = 434 // patch_size
        width = 626 // patch_size

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channel=in_channels,
            embed_dim=emb_size,
            kernel_size=patch_size,
            stride=patch_size
        )

        print(height // 2, width // 2)
        # Swin Transformer blocks (first stage)
        self.swin_blocks_1 = nn.ModuleList([
            SwinTransformerBlock(
                dim=emb_size*6,
                input_resolution=(height, width),
                num_heads=num_heads[1],
                window_size=swin_window_size[1],
                shift_size=0 if i % 2 == 0 else swin_window_size[1] // 2
            )
            for i in range(swin_blocks[1])
        ])

        # Step 4: Single Channel Attention Module
        #self.cam_2 = CAM(in_channels=2*emb_size, reduction_ratio=reduction_ratio)
        self.patch_merging_2 = PatchMerging(emb_size*6)

        # Step 6: Swin Transformer Blocks (Second Group, 1 block)
        self.swin_blocks_2 = nn.ModuleList([
            SwinTransformerBlock(
                dim=6*2*emb_size,
                input_resolution=(height//2, width//2),
                num_heads=num_heads[2],
                window_size=swin_window_size[2],
                shift_size=0 if i % 2 == 0 else swin_window_size[2] // 2
            )
            for i in range(swin_blocks[2])
        ])

        # Step 7: Global Pooling and Fully Connected Layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(6*2*emb_size, 512),
            nn.ELU(),
            nn.Linear(512, 1)
        )

    def forward(self, x_sai, x_mli):
      

        print(x_sai.shape)
        print(x_mli.shape)
        x_sai = x_sai.permute(0, 2, 1, 3, 4)
        x = self.patch_embed(x_sai)  # Shape: [batch_size, num_patches, emb_size]
        x = rearrange(x, 'b c h w -> b h w c')   
        print(x.shape)
        # Pass through Swin blocks
        for block in self.swin_blocks_1:
            x = block(x)

        print(x.shape)
        x = self.patch_merging_2(x)
        print(x.shape)
        for swin_block in self.swin_blocks_2:
            x = swin_block(x)  
        print(x.shape)

        van_layer1, _, _, _ = self.van(x_mli) 
        print(van_layer1.shape)

        # Step 7: Global Pooling and Fully Connected Layer
        x = rearrange(x, 'b h w c -> b c h w')  # Restore to [batch_size, emb_size, height, width]
        x = self.global_pool(x)  # Reduce spatial dimensions to [batch_size, emb_size, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, emb_size]
        x = self.fc(x)  # Map to [batch_size, 1]

        return x
