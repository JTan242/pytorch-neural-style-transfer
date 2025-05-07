from collections import namedtuple
import torch
from torchvision import models

class AlexNet(torch.nn.Module):
    """
    Feature extractor based on AlexNet for neural style transfer.
    Uses conv1 through conv5 to extract content and style features.
    """
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        alexnet = models.alexnet(pretrained=True, progress=show_progress)
        features = alexnet.features

        # Layer slices
        self.slice0 = torch.nn.Sequential(*features[0:1])   
        self.slice1 = torch.nn.Sequential(*features[1:3])  
        self.slice2 = torch.nn.Sequential(*features[3:6])   
        self.slice3 = torch.nn.Sequential(*features[6:8])   
        self.slice4 = torch.nn.Sequential(*features[8:10]) 
        self.slice5 = torch.nn.Sequential(*features[10:12]) 
        self.slice6 = torch.nn.Sequential(*features[12:14]) 
        self.slice7 = torch.nn.Sequential(*features[14:]) 

        self.layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        self.content_feature_maps_index = 4  
        self.style_feature_maps_indices = list(range(5)) 

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice0(x)
        conv1 = x
        x = self.slice1(x)
        x = self.slice2(x)
        conv2 = x
        x = self.slice3(x)
        x = self.slice4(x)
        conv3 = x
        x = self.slice5(x)
        conv4 = x
        x = self.slice6(x)
        conv5 = x
        _ = self.slice7(x)  

        alexnet_outputs = namedtuple("AlexNetOutputs", self.layer_names)
        return alexnet_outputs(conv1, conv2, conv3, conv4, conv5)
