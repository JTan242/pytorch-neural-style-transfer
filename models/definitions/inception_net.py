from torchvision import models
import torch
from collections import namedtuple

class InceptionNet(torch.nn.Module):
    """
    Improved feature extractor based on Inception v3 for neural style transfer.
    Uses both early and deep layers to better capture style and content.
    """
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
        if show_progress:
            print("Loaded pretrained Inception v3")

        self.stem = torch.nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.mixed_5b = inception.Mixed_5b
        self.mixed_5c = inception.Mixed_5c
        self.mixed_5d = inception.Mixed_5d
        self.mixed_6a = inception.Mixed_6a
        self.mixed_6b = inception.Mixed_6b
        self.mixed_6c = inception.Mixed_6c
        self.mixed_6d = inception.Mixed_6d
        self.mixed_6e = inception.Mixed_6e
        self.mixed_7a = inception.Mixed_7a
        self.mixed_7b = inception.Mixed_7b
        self.mixed_7c = inception.Mixed_7c

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        # Output layers
        self.layer_names = ['Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6e', 'Mixed_7c']
        self.content_feature_maps_index = 3  
        self.style_feature_maps_indices = list(range(len(self.layer_names)))

    def forward(self, x):
        x = self.stem(x)
        x = self.mixed_5b(x)
        feat_5b = x
        x = self.mixed_5c(x)
        feat_5c = x
        x = self.mixed_5d(x)
        feat_5d = x
        x = self.mixed_6a(x)
        x = self.mixed_6b(x)
        x = self.mixed_6c(x)
        x = self.mixed_6d(x)
        x = self.mixed_6e(x)
        feat_6e = x
        x = self.mixed_7a(x)
        x = self.mixed_7b(x)
        x = self.mixed_7c(x)
        feat_7c = x

        outputs = namedtuple("InceptionOutputs", self.layer_names)
        return outputs(feat_5b, feat_5c, feat_5d, feat_6e, feat_7c)
