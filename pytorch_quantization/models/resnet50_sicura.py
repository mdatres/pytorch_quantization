from . import ResNet50
from typing import Any, List, Optional, Type, Union
from torchvision.models.quantization import  ResNet50_QuantizedWeights
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.quantization import QuantizableResNet
import torch.nn as nn

class ResNet50_sicura(ResNet50):

    def __init__(self, weights: Optional[Union[ResNet50_QuantizedWeights, ResNet50_Weights]] = None, progress: bool = True, quantize: bool = False, **kwargs: Any) -> QuantizableResNet:
        super().__init__(weights, progress, quantize, **kwargs)
        in_features = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Linear(in_features, 500), nn.ReLU(), nn.Dropout(), nn.Linear(500, 2))