from . import ResNet34
from typing import Any, List, Optional
from torchvision.models.resnet import ResNet34_Weights
from torchvision.models.quantization import QuantizableResNet
import torch.nn as nn

class ResNet34_sicura(ResNet34):

    def __init__(self, weights: Optional[ResNet34_Weights] = None, progress: bool = True, quantize: bool = False, **kwargs: Any) -> QuantizableResNet:
        super().__init__(weights, progress, quantize, **kwargs)
        in_features = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Linear(in_features, 500), nn.ReLU(), nn.Dropout(), nn.Linear(500, 2))