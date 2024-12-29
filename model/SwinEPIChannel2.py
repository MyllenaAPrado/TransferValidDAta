import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock, PatchMerging
from einops import rearrange
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


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
    def __init__(self, image_size, in_channels, patch_size, emb_size, reduction_ratio, swin_window_size, num_heads, swin_blocks, num_stb, size_input):
        super(IntegratedModelV2, self).__init__()

        # MobileNetV3-Small as feature extractor
        mobilenet = mobilenet_v3_small(pretrained=True)
        self.feature_extractor = mobilenet.features

        # Efficient Channel Attention (ECA)
        self.eca = ECA(channels=16)  # Shallow features have 16 channels

        # Spatial Attention Module
        self.spatial_attention = SpatialAttention(kernel_size=7)  # For deeper features

        # Patch embedding
        self.patch_embedding = nn.Conv2d(64, emb_size, kernel_size=patch_size, stride=patch_size)

        # Swin Transformer blocks
        height = size_input[0]//patch_size#44
        width = size_input[1]//patch_size #5
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=emb_size,
                input_resolution=(height, width),
                num_heads=num_heads[0],
                window_size=swin_window_size[0],
                shift_size=0 if i % 2 == 0 else swin_window_size[0] // 2
            )
            for i in range(swin_blocks[0])
        ])

        self.patch_merging_1 = PatchMerging(emb_size)        
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
    
        # Global pooling and fully connected layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.num_stb = num_stb
        if self.num_stb == 1:
            features = 1
        else:
            features = 2

        self.fc = nn.Sequential(
            nn.Linear(features*emb_size, 512),
            nn.ELU(),
            nn.Linear(512, 1)
        )


    def forward(self, x):
        # Extract shallow and deep features
        shallow_features = self.feature_extractor[:3](x)  # Early layers
        deep_features = self.feature_extractor[3:6](shallow_features)   # Later layers

        # Apply ECA to shallow features
        refined_shallow = self.eca(shallow_features)

        # Apply Spatial Attention to deep features
        refined_deep = self.spatial_attention(deep_features)

        # Align dimensions (if necessary)
        refined_shallow = F.interpolate(refined_shallow, size=refined_deep.size()[2:], mode='bilinear', align_corners=False)

        # Fuse the features (concatenate or add)
        fused_features = torch.cat([refined_shallow, refined_deep], dim=1)

        # Apply patch embedding
        x = self.patch_embedding(fused_features)

        # Rearrange for Swin Transformer
        x = rearrange(x, 'b c h w -> b h w c')

        # Pass through Swin Transformer blocks
        for swin_block in self.swin_blocks:
            x = swin_block(x)

        if self.num_stb == 2:
            x = self.patch_merging_1(x)
            for swin_block in self.swin_blocks_1:
                x = swin_block(x)  

        # Global Pooling and Fully Connected Layer
        x = rearrange(x, 'b h w c -> b c h w')  # Restore to [batch_size, emb_size, height, width]
        x = self.global_pool(x)  # Reduce spatial dimensions to [batch_size, emb_size, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, emb_size]
        #print(x.shape)

        x = self.fc(x)  # Fully connected output

        return x



'''class IntegratedModelV2(nn.Module):
    def __init__(self, image_size, in_channels, patch_size, emb_size, reduction_ratio, swin_window_size, num_heads, swin_blocks,
                num_stb):
        super(IntegratedModelV2, self).__init__()

        # Use pre-trained MobileNetV3-Small for feature extraction
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        self.extract_features = nn.Sequential(
            mobilenet.features,  # Extract features from MobileNetV3
            #nn.Conv2d(576, emb_size, kernel_size=1)  # Reduce channels to match emb_size
        )
        
        self.num_stb = num_stb       
        self.patch_embedding = nn.Conv2d(576, emb_size, kernel_size=patch_size, stride=patch_size)
        
        height = 105 // patch_size
        width = 16 // patch_size
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
        
        self.cam_1 = CAM(in_channels=emb_size, reduction_ratio=reduction_ratio)
        self.patch_merging_1 = PatchMerging(emb_size)        
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
        
        self.cam_2 = CAM(in_channels=2*emb_size, reduction_ratio=reduction_ratio)
        self.patch_merging_2 = PatchMerging(emb_size*2)

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
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(features*emb_size, 512),
            nn.ELU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        
        x = self.extract_features(x)
        #print(x.shape)
        
        #print(low_features.shape) 
        #high_features = self.feature_extractor(x)  
        # Resize low_features to match the spatial size of high_features
        #low_features = F.interpolate(low_features, size=high_features.shape[2:], mode='bilinear', align_corners=False)
        #combined_features = torch.cat([low_features, high_features], dim=1)

        x = self.patch_embedding(x) 
        #print(x.shape)

        x = rearrange(x, 'b c h w -> b h w c')   
        #print(x.shape)

        for swin_block in self.swin_blocks_:
            x = swin_block(x) 

        if self.num_stb == 2 or self.num_stb == 3:
            
            x = rearrange(x, 'b h w c -> b c h w')  
            x = self.cam_1(x)  # Apply channel attention
            x = rearrange(x, 'b c h w -> b h w c')  

            x = self.patch_merging_1(x)
            for swin_block in self.swin_blocks_1:
                x = swin_block(x)  

        if self.num_stb == 3:
            x = rearrange(x, 'b h w c -> b c h w')  
            x = self.cam_2(x)  # Apply channel attention
            x = rearrange(x, 'b c h w -> b h w c') 
            
            x = self.patch_merging_2(x)
            for swin_block in self.swin_blocks_2:
                x = swin_block(x)  

        # Global Pooling and Fully Connected Layer
        x = rearrange(x, 'b h w c -> b c h w')  # Restore to [batch_size, emb_size, height, width]
        x = self.global_pool(x)  # Reduce spatial dimensions to [batch_size, emb_size, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, emb_size]
        x = self.fc(x)  

        return x
'''