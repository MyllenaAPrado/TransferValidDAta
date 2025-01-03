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
#from model.NAT import nat_mini
import timm


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

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

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

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.save_output = SaveOutput()

        # Freeze all layers
        for param in self.vit.parameters():
            param.requires_grad = False

        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        self.num_features=4

        self.eca = eca_layer()

        self.avg_pool = nn.AdaptiveAvgPool2d(2)
        self.rerange_layer = Rearrange('b c h w -> b (h w) c')

        self.patch_embedding = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)


        # Patch embedding
        self.patch_embedding = nn.Conv2d(192 *self.num_features, 128, kernel_size=3, stride=3)


        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=128,
                input_resolution=(23, 23),
                num_heads=2,
                window_size=swin_window_size[0],
                shift_size=0 if i % 2 == 0 else swin_window_size[0] // 2
            )
            for i in range(2)
        ])
        # Step 4: Single Channel Attention Module
        self.patch_merging_1 = PatchMerging(128)
        # Step 6: Swin Transformer Blocks (Second Group, 1 block)
        self.swin_blocks_1 = nn.ModuleList([
            SwinTransformerBlock(
                dim=2*128,
                input_resolution=(23//2, 23//2),
                num_heads=num_heads[1],
                window_size=swin_window_size[1],
                shift_size=0 if i % 2 == 0 else swin_window_size[1] // 2
            )
            for i in range(swin_blocks[1])
        ])

        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.Linear( 1024, 512),  
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x



    def forward(self, x_sai, x_mli):
        batch_size, _, _, _,_ = x_sai.shape

        x_sai = x_sai.reshape(batch_size*25, 3, 224, 224) 

        x = self.vit(x_sai)  # Features shape: (batch_size * 16, embed_dim)
        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()

        x = x.view(batch_size, 25, 192 *self.num_features, 28, 28)
        x = x.view(batch_size, 5, 5, 192 *self.num_features, 28, 28)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, 192 *self.num_features, 5 * 28, 5 * 28)

        x = self.patch_embedding(x)

        print(x.shape)

        # Step 3: First Group of Swin Transformer Blocks
        x = rearrange(x, 'b c h w -> b h w c')  # Rearrange for Swin Transformer
        for swin_block in self.swin_blocks:
            x = swin_block(x) 

        x = self.patch_merging_1(x)
        for swin_block in self.swin_blocks_1:
            x = swin_block(x)  
        x = rearrange(x, 'b h w c -> b c h w')  # Restore to [batch_size, emb_size, height, width]
        print(x.shape)

        x = self.eca(x)
        print(x.shape)

        x = self.avg_pool(x)
        print(x.shape)

        return self.regression(x)
