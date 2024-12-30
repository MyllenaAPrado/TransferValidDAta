import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock, PatchMerging
from einops import rearrange
import torch.nn as nn
import torchvision.models as models

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg
from model.VAN import van_b2, van_b1  # Adjust import based on where VAN is defined
import math
import torch.nn.functional as F
from einops.layers.torch import Rearrange

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


class IntegratedModelV2(nn.Module):
    def __init__(self, image_size, in_channels, patch_size, emb_size, reduction_ratio, swin_window_size, num_heads, swin_blocks,
                num_stb):
        super(IntegratedModelV2, self).__init__()

        self.conv_down = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=12,
            stride=12
        )

        self.van = van_b1(pretrained=True)  # Pretrained VAN model (van_b0 or other variant)     

        self.eca = ECA3DLayer()
        self.avg_pool = nn.AdaptiveAvgPool2d(224 // 32)
        self.rerange_layer = Rearrange('b c h w -> b (h w) c')

         # Patch embedding
        self.patch_embedding = nn.Conv2d(3, 32, kernel_size=48, stride=48)

        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=32,
                input_resolution=(224, 224),
                num_heads=2,
                window_size=swin_window_size[0],
                shift_size=0 if i % 2 == 0 else swin_window_size[0] // 2
            )
            for i in range(1)
        ])

        embed_dim = 1056
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


    def forward(self, x_sai, x_mli):
        #print(x_mli.shape)
        #print(x_sai.shape)
        batch, _, _, _,_ =x_sai.shape
        x_mli=self.conv_down(x_mli)
        layer1_s, layer2_s, layer3_s, layer4_s = self.van(x_mli)    # (b,64,56,56); (b,128,28,28); (b,320,14,14); (b,512,7,7)
        s1 = self.avg_pool(layer1_s)
        s2 = self.avg_pool(layer2_s)
        s3 = self.avg_pool(layer3_s)
        s4 = self.avg_pool(layer4_s)

        print(s1.shape)
        print(s2.shape)
        print(s3.shape)
        print(s4.shape)


        #print(x_sai.shape)
        x_sai = self.eca(x_sai)
        x_sai = x_sai.reshape(batch, 5, 5, 3, 434, 626)  # [batch_size, grid_h, grid_w, channels, height, width]
        x_sai = x_sai.permute(0, 3, 1, 4, 2, 5)  # [batch_size, channels, grid_h, height, grid_w, width]
        x_sai = x_sai.reshape(batch, 3, 5 * 434, 5 * 626)  # [batch_size, channels, total_height, total_width]
        x_sai = self.patch_embedding(x_sai)
        x_sai = rearrange(x_sai, 'b c h w -> b h w c')
        # Pass through Swin Transformer blocks
        for swin_block in self.swin_blocks:
            x_sai = swin_block(x_sai)
        x_sai = rearrange(x_sai, 'b h w c-> b c h w')  
        x_sai = self.avg_pool(x_sai)
        print(x_mli.shape)
        print(x_sai.shape)

        feats = torch.cat((s1, s2, s3, s4, x_sai), dim=1)
        feats = self.rerange_layer(feats)  # (b, c, h, w) -> (b, h*w, c)

        scores = self.head_score(feats)
        weights = self.head_weight(feats)
        q = torch.sum(scores * weights, dim=1) / torch.sum(weights, dim=1)

        return q
