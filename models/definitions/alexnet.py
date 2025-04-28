from collections import namedtuple
import torch
from torchvision import models

class AlexNet(torch.nn.Module):
    """
    Feature extractor based on AlexNet for neural style transfer.
    Only uses deeper layers (conv3, conv4, conv5) for style and content features.
    """
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        alexnet = models.alexnet(pretrained=True, progress=show_progress)
        features = alexnet.features
        
        # Slices based on AlexNet architecture
        self.slice1 = torch.nn.Sequential(*features[0:3])   # conv1 + relu
        self.slice2 = torch.nn.Sequential(*features[3:6])   # lrn + maxpool + conv2
        self.slice3 = torch.nn.Sequential(*features[6:8])   # relu + lrn
        self.slice4 = torch.nn.Sequential(*features[8:10])  # maxpool + conv3
        self.slice5 = torch.nn.Sequential(*features[10:12]) # relu + conv4
        self.slice6 = torch.nn.Sequential(*features[12:14]) # relu + conv5
        self.slice7 = torch.nn.Sequential(*features[14:])   # relu + maxpool

        # Define the output feature maps you will use
        self.layer_names = ['conv3', 'conv4', 'conv5']
        self.content_feature_maps_index = 0  # conv3 for content
        self.style_feature_maps_indices = list(range(len(self.layer_names)))  # conv3, conv4, conv5 for style

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)  # conv1 + relu
        x = self.slice2(x)  # lrn + maxpool + conv2
        x = self.slice3(x)  # relu + lrn
        x = self.slice4(x)  # maxpool + conv3
        conv3 = x           # Save conv3 output
        x = self.slice5(x)  # relu + conv4
        conv4 = x           # Save conv4 output
        x = self.slice6(x)  # relu + conv5
        conv5 = x           # Save conv5 output
        _ = self.slice7(x)  # Final maxpool (not used)

        # Return only the selected features
        alexnet_outputs = namedtuple("AlexNetOutputs", self.layer_names)
        return alexnet_outputs(conv3, conv4, conv5)