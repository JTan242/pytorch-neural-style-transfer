from torchvision import models
import torch
from collections import namedtuple

class InceptionNet(torch.nn.Module):
    """
    Feature extractor based on Inception v3 for neural style transfer.
    Only outputs mid-to-deep layer activations useful for style and content representation.
    """
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True, transform_input=False)
        if show_progress:
            print("Loaded pretrained Inception v3")

        # We'll use key blocks: Mixed_5d, Mixed_6e, Mixed_7c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7c = inception.Mixed_7c

        # Early feature extractors (stem)
        self.stem = torch.nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
        )

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.layer_names = ['Mixed_5d', 'Mixed_6e', 'Mixed_7c']
        self.content_feature_maps_index = 1  # e.g., Mixed_6e
        self.style_feature_maps_indices = list(range(len(self.layer_names)))

    def forward(self, x):
        x = self.stem(x)
        x = self.Mixed_5d(x)
        feat_5d = x
        x = self.Mixed_6e(x)
        feat_6e = x
        x = self.Mixed_7c(x)
        feat_7c = x

        outputs = namedtuple("InceptionOutputs", self.layer_names)
        return outputs(feat_5d, feat_6e, feat_7c)
