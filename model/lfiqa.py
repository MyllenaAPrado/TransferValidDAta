import torch
import torch.nn as nn
import timm
import numpy as np
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from einops import rearrange
from model.swinOrig import SwinTransformer

class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []


class LFIQAVSTModel(nn.Module):
    def __init__(self, depths= [2, 2, 2],
                 num_fcc1 =1024, num_fcc2=512, num_fcc3=64):
        super(LFIQAVSTModel, self).__init__()
        
        self.vit =  timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.save_output = SaveOutput()

        # Freeze all layers
        for param in self.vit.parameters():
            param.requires_grad = False

        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        
        
        self.num_features = 4
        embed_dim = 256

        self.conv = nn.Conv2d(768 * self.num_features, embed_dim, 1, 1, 0)
        self.patches_resolution =(5*28, 5*28)

        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=[2, 2, 2],
            num_heads=[4, 4, 4],
            embed_dim=embed_dim,
            window_size=10,
            dim_mlp=256,
            scale=0.2
        )
        self.conv2 = nn.Conv2d(embed_dim, embed_dim//4, 2, 2, 0)

        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.Linear((256//4)*(140//2)*(140//2) , num_fcc1),  
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(num_fcc1, num_fcc2),
            nn.ELU(),
            nn.Linear(num_fcc2, num_fcc3),
            nn.ELU(),
            nn.Linear(num_fcc3, 1)
        )

    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def forward(self, x):

        batch_size = x.size(0)
        # x is expected to be of shape (batch_size, 4, 4, 3, 224, 224)
        x = x.view(batch_size, 25, 3, 224, 224)  
        # Flatten batch for processing through ViT
        x = x.view(batch_size * 25, 3, 224, 224)  

        # Extract features using ViT
        x = self.vit(x)  # Features shape: (batch_size * 16, embed_dim)
        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()

        #reshape data
        x = x.view(batch_size, 25, 768 *self.num_features, 28, 28)
        x = x.view(batch_size, 5, 5, 768 *self.num_features, 28, 28)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, 768 *self.num_features, 5 * 28, 5 * 28)

        #apply swin block
        x = self.conv(x) 
        x = self.swintransformer1(x) 
        x = self.conv2(x) 
        #print(x.shape)
        ## Ensure the output is flattened correctly
        x = x.view(batch_size, -1)  # Flatten the output for the regression layer
        x = self.regression(x)

        return x
