from collections import namedtuple
import torch
from torchvision import models

class AlexNet(torch.nn.Module):
    """
    Feature extractor based on AlexNet.
    This version splits the network into sequential slices and returns selected intermediate feature maps
    for neural style transfer purposes.
    """
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        alexnet = models.alexnet(pretrained=True, progress=show_progress)
        features = alexnet.features
        
        # AlexNet layers: conv1 -> relu -> lrn -> maxpool -> conv2 -> relu -> lrn -> maxpool -> conv3 -> relu -> conv4 -> relu -> conv5 -> relu -> maxpool
        # We'll extract output from conv1, conv2, conv3, conv4, and conv5 for style/content features
        self.layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        self.content_feature_maps_index = 2  # conv3
        self.style_feature_maps_indices = list(range(len(self.layer_names)))

        self.slice1 = torch.nn.Sequential(*features[0:3])   # conv1 + relu
        self.slice2 = torch.nn.Sequential(*features[3:6])   # lrn + maxpool + conv2
        self.slice3 = torch.nn.Sequential(*features[6:8])   # relu + lrn
        self.slice4 = torch.nn.Sequential(*features[8:10])  # maxpool + conv3
        self.slice5 = torch.nn.Sequential(*features[10:12]) # relu + conv4
        self.slice6 = torch.nn.Sequential(*features[12:14]) # relu + conv5
        self.slice7 = torch.nn.Sequential(*features[14:])   # relu + maxpool

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        conv1 = x
        x = self.slice2(x)
        x = self.slice3(x)
        conv2 = x
        x = self.slice4(x)
        conv3 = x
        x = self.slice5(x)
        conv4 = x
        x = self.slice6(x)
        conv5 = x
        _ = self.slice7(x)  # Not used, but completes the forward pass

        alexnet_outputs = namedtuple("AlexNetOutputs", self.layer_names)
        return alexnet_outputs(conv1, conv2, conv3, conv4, conv5)
