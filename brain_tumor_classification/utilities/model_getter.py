import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


def get_model(n_unfrozen, out_features) -> torch.nn.Module:
    """Get model to work with"""

    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(in_features, out_features)

    for param in model.features.parameters():
        param.requires_grad = False

    for param in model.features[-n_unfrozen:].parameters():
        param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model
