from typing import Any, List, Optional, Type, Union
import torch.nn as nn
from torchvision.models.quantization import resnet18, resnet50, resnext101_32x8d, resnext101_64x4d, QuantizableResNet
from torchvision.models.quantization.resnet import _resnet, QuantizableBasicBlock, QuantizableBottleneck
from torchvision.models.quantization import ResNet18_QuantizedWeights, ResNet50_QuantizedWeights, ResNeXt101_32X8D_QuantizedWeights, ResNeXt101_64X4D_QuantizedWeights
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet34_Weights,
    ResNet152_Weights, 
    ResNeXt101_32X8D_Weights,
    ResNeXt101_64X4D_Weights,
)
__all__ = [
    "QuantizableResNet",
    "ResNet18_QuantizedWeights",
    "ResNet50_QuantizedWeights",
    "ResNeXt101_32X8D_QuantizedWeights",
    "ResNeXt101_64X4D_QuantizedWeights",
    "resnet18",
    "resnet50",
    "resnext101_32x8d",
    "resnext101_64x4d",
]
class ResNet18(nn.Module):

    def __init__(*, weights: Optional[Union[ResNet18_QuantizedWeights, ResNet18_Weights]] = None, progress: bool = True, quantize: bool = False, **kwargs: Any)-> QuantizableResNet:
        super().__init__()
        return resnet18(weights = weights, progress = progress, quantize = quantize, **kwargs)

class ResNet34(nn.Module):

    def __init__(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, quantize: bool = False, **kwargs: Any)-> QuantizableResNet:
        super().__init__()
        weights = (ResNet34_Weights).verify(weights)
        return _resnet(QuantizableBasicBlock, [3, 4, 6, 3], weights, progress, quantize, **kwargs)



class ResNet50(nn.Module):

    def __init__(*, weights: Optional[Union[ResNet50_QuantizedWeights, ResNet50_Weights]] = None, progress: bool = True, quantize: bool = False, **kwargs: Any)-> QuantizableResNet:
        super().__init__()
        return resnet50(weights = weights, progress = progress, quantize = quantize, **kwargs)

class ResNet152(nn.Module):

    def __init__(*, weights: Optional[ResNet152_Weights] = None, progress: bool = True, quantize: bool = False, **kwargs: Any)-> QuantizableResNet:
        super().__init__()
        weights = ResNet152_Weights.verify(weights)

        return _resnet(QuantizableBottleneck, [3, 8, 36, 3], weights = weights, progress = progress, quantize = quantize, **kwargs)

class ResNeXt101_32X8D(nn.Module):

    def __init__(*, weights: Optional[Union[ResNeXt101_32X8D_QuantizedWeights, ResNeXt101_32X8D_Weights]] = None, progress: bool = True, quantize: bool = False, **kwargs: Any)-> QuantizableResNet:
        super().__init__()
        return resnext101_32x8d(weights = weights, progress = progress, quantize = quantize, **kwargs)


class ResNeXt101_64X4D(nn.Module):

    def __init__(*, weights: Optional[Union[ResNeXt101_64X4D_QuantizedWeights, ResNeXt101_64X4D_Weights]] = None, progress: bool = True, quantize: bool = False, **kwargs: Any)-> QuantizableResNet:
        super().__init__()
        return resnext101_64x4d(weights = weights, progress = progress, quantize = quantize, **kwargs)
