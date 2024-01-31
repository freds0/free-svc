import torch
import torch.nn as nn
from typing import Union, Callable, List, Optional
from torchvision.models import efficientnet_b0, efficientnet_b4
import torchvision.transforms as vt


def get_vision_backbone(
    vismod="efficientnet_b0", num_classes=1000, pretrained=False, **kwargs
):
    if vismod == "efficientnet_b0":
        return efficientnet_b0(pretrained=pretrained, num_classes=num_classes, **kwargs)
    elif vismod == "efficientnet_b4":
        return efficientnet_b4(pretrained=pretrained, num_classes=num_classes, **kwargs)

    else:
        raise NotImplementedError


class Grey2Rgb(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = vt.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, data):
        batch_size, freq_bins, times = data.shape
        data /= data.max()
        data = data.unsqueeze(1).expand(batch_size, 3, freq_bins, times)
        data = self.normalize(data)
        return data


class LogScale(nn.Module):
    def forward(self, data):
        # eps = 1e-8
        eps = torch.tensor(1e-8, device=data.device)
        return torch.log(data + eps)


class Aggregator(nn.Module):
    """Aggregates (in time) a list of features"""

    def __init__(self):
        super().__init__()
        self.aggregation = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(1))

    def forward(self, features):
        """
        Returns:
            outputs_feature: torch.Tensor of shape(B x C x t)
        """
        if isinstance(features, list):
            output_feature = [self.aggregation(feature) for feature in features]
        else:
            output_feature = self.aggregation(features)
        return output_feature
