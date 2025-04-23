# Resnet-50 network modified for NST (Neural Style Transfer):

import torch
from torch import nn
from torchvision import models
from collections import namedtuple

'''class ResNet50_NST(nn.Module):
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()

        self.layer_names = ['stage1', 'stage2', 'stage3', 'stage4']
        self.content_feature_maps_index = 2   # use slice3 for content
        self.style_feature_maps_indices  = [0, 1, 2, 3] # use all slices for style, except for slice3
        self.style_feature_maps_indices.remove()(2) # remove slice3 for style'''
class ResNet50(nn.Module):
    """
    Exposes intermediate ResNet_50 activations for style/content.
    
    Style layers: 
      conv1 -> layer1 -> layer2 -> layer3 -> layer4
    Content layer: 
      layer3
    """
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        # Load pretrained ResNet‑50
        # https://pytorch.org/vision/stable/models.html#id1
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        resnet = models.resnet50(pretrained=True, progress=show_progress)
        

        self.conv1   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        

        self.layer_names = [
            'conv1', 
            'layer1',  
            'layer2',  
            'layer3',  # content features
            'layer4',  # deepest features
        ]
        self.content_feature_maps_index = 3  # 'layer3'
        
        # all others become style layers
        self.style_feature_maps_indices = list(range(len(self.layer_names)))
        self.style_feature_maps_indices.remove(self.content_feature_maps_index)
        
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        # normalize just once up front
        x = (x - self.mean) / self.std
        out_conv1  = self.conv1(x)       # [B,  64, H/2, W/2]
        x = self.maxpool(out_conv1)
        out_layer1 = self.layer1(x)      # [B, 256, H/4, W/4]
        out_layer2 = self.layer2(out_layer1)  # [B, 512, H/8, W/8]
        out_layer3 = self.layer3(out_layer2)  # [B,1024, H/16,W/16]
        out_layer4 = self.layer4(out_layer3)  # [B,2048, H/32,W/32]

        ResNetOutputs = namedtuple("ResNetOutputs", self.layer_names)
        return ResNetOutputs(
            out_conv1,
            out_layer1,
            out_layer2,
            out_layer3,
            out_layer4,
        )