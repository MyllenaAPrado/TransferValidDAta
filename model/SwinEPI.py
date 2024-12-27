import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock, PatchMerging
from einops import rearrange

class ExtractLowFeaturesWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExtractLowFeaturesWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        edge_map = self.sigmoid(self.conv2(x))        
        
        return x * edge_map 

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
        # Step 1: Extract Low Features with Attention
        if EAM:
            self.eam = ExtractLowFeaturesWithAttention(in_channels=in_channels, out_channels=in_channels)
        else:
            self.eam = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, in_channels, kernel_size=1)
            )

        
        self.num_stb = num_stb

        # Step 2: Patch Embedding
        self.patch_embedding = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        
        # Step 3: Swin Transformer Blocks (First Group, 1 block)
        height = image_size[0] // patch_size
        width = image_size[1] // patch_size
        self.swin_blocks_ = nn.ModuleList([
            SwinTransformerBlock(
                dim=emb_size,
                input_resolution=(height, width),
                num_heads=num_heads[0],
                window_size=swin_window_size[0],
                shift_size=0 if i % 2 == 0 else swin_window_size[0] // 2
            )
            for i in range(swin_blocks[0])
        ])
        
        # Step 4: Single Channel Attention Module
        self.patch_merging_1 = PatchMerging(emb_size)
        # Step 6: Swin Transformer Blocks (Second Group, 1 block)
        self.swin_blocks_1 = nn.ModuleList([
            SwinTransformerBlock(
                dim=2*emb_size,
                input_resolution=(height//2, width//2),
                num_heads=num_heads[1],
                window_size=swin_window_size[1],
                shift_size=0 if i % 2 == 0 else swin_window_size[1] // 2
            )
            for i in range(swin_blocks[1])
        ])

        # Step 4: Single Channel Attention Module
        self.patch_merging_2 = PatchMerging(emb_size*2)

        # Step 6: Swin Transformer Blocks (Second Group, 1 block)
        self.swin_blocks_2 = nn.ModuleList([
            SwinTransformerBlock(
                dim=4*emb_size,
                input_resolution=(height//4, width//4),
                num_heads=num_heads[2],
                window_size=swin_window_size[2],
                shift_size=0 if i % 2 == 0 else swin_window_size[2] // 2
            )
            for i in range(swin_blocks[2])
        ])

        if self.num_stb == 1:
            features = 1
        elif self.num_stb == 2:
            features = 2
        else:
            features = 4
        
        # Step 7: Global Pooling and Fully Connected Layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(features*emb_size, 512),
            nn.ELU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # Step 1: Extract Low Features
        x = self.eam(x)  # Enhance edges: [batch_size, 3, height, width]

        # Step 2: Initial Patch Embedding
        x = self.patch_embedding(x)  # Convert to patches: [batch_size, emb_size, height/patch_size, width/patch_size]

        # Step 3: First Group of Swin Transformer Blocks
        x = rearrange(x, 'b c h w -> b h w c')  # Rearrange for Swin Transformer
        for swin_block in self.swin_blocks_:
            x = swin_block(x) 

        if self.num_stb == 2 or self.num_stb ==3:
            x = self.patch_merging_1(x)
            for swin_block in self.swin_blocks_1:
                x = swin_block(x)  

        if  self.num_stb == 3:            
            x = self.patch_merging_2(x)
            for swin_block in self.swin_blocks_2:
                x = swin_block(x)  

        # Step 7: Global Pooling and Fully Connected Layer
        x = rearrange(x, 'b h w c -> b c h w')  # Restore to [batch_size, emb_size, height, width]
        x = self.global_pool(x)  # Reduce spatial dimensions to [batch_size, emb_size, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, emb_size]
        x = self.fc(x)  # Map to [batch_size, 1]

        return x
