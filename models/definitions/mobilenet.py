
# utils/mobilenet_nst.py

import torch
from torch import nn
from torchvision import models
from collections import namedtuple

class MobileNetV2_NST(nn.Module):
    """
    MobileNetV2 wrapped for Neural Style Transfer:
     - normalizes input
     - chops features into four “stages”
     - exposes .layer_names, .content_feature_maps_index,
       and .style_feature_maps_indices for your NST loop
    """
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()

        # Normalize with ImageNet mean and std
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        # Pick out the layers to use
        self.layer_names = ['stage1','stage2','stage3','stage4']
        self.content_feature_maps_index = 2   # use slice3 for content
        self.style_feature_maps_indices  = [0,1,2,3]

        # Create the slices
        feats = models.mobilenet_v2(pretrained=True, progress=show_progress).features
        self.slice1 = nn.Sequential(*feats[0:3])   # edges
        self.slice2 = nn.Sequential(*feats[3:5])   # textures
        self.slice3 = nn.Sequential(*feats[5:9])   # patterns
        self.slice4 = nn.Sequential(*feats[9:19])  # semantics

        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        # normalize just once up front
        x = (x - self.mean) / self.std

        # run through the four slices
        s1 = self.slice1(x)
        s2 = self.slice2(s1)
        s3 = self.slice3(s2)
        s4 = self.slice4(s3)

        # stick them into a namedtuple so indexing code works
        MobileOutputs = namedtuple("MobileOutputs", self.layer_names)
        return MobileOutputs(s1, s2, s3, s4)
