import torch
from torch import nn
from torchvision import models
from collections import namedtuple

class CNN(nn.Module):
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()

        # Create layers of CNN
        self.layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        self.content_feature_maps_index = 4
        self.style_feature_maps_indices = [0, 1, 2, 3, 4]

        # First Layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second layer
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        #Third layer
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        #Fourth layer
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        #Fifth layer
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(256 * 7 * 7, 10)
        
        # Normalize with ImageNet mean and std
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Normalize
        x = (x - self.mean) / self.std

        # Forward pass through CNN layers
        layer1 = self.pool1(self.relu1(self.conv1(x)))
        layer2 = self.pool2(self.relu2(self.conv2(layer1)))
        layer3 = self.pool3(self.relu3(self.conv3(layer2)))
        layer4 = self.pool4(self.relu4(self.conv4(layer3)))
        layer5 = self.pool5(self.relu5(self.conv5(layer4)))

        # namedtuple so indexing code works
        CNNOutputs = namedtuple("CNNOutputs", self.layer_names)
        return CNNOutputs(layer1, layer2, layer3, layer4, layer5)