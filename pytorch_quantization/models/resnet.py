
from numpy import dtype
import torch
import torch.nn as  nn
import torch.nn.functional as F
from typing import NamedTuple, List, Union, Optional, Tuple




import torch
import torch.nn as  nn
import torch.nn.functional as F




class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        
        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if self.downsample else 1, bias = False),
                nn.BatchNorm2d(out_channels)
            )

        
        
       

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = self.relu1(self.bn1(self.conv1(input)))
        input = self.relu2(self.bn2(self.conv2(input)))
        input = self.relu3(self.bn3(self.conv3(input)))
        #input = input + shortcut
        input = self.skip_add.add(input, shortcut)
        return self.relu4(input)


class Resnet(nn.Module):
    def __init__(self, in_channels, resblock, repeat, useBottleneck=False, outputs=2):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.skip_add = nn.quantized.FloatFunctional()
        
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', resblock(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
                self.layer1.add_module('conv2_%d'%(i+1,), resblock(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', resblock(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
                self.layer2.add_module('conv3_%d' % (i+1,), resblock(filters[2], filters[2], downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', resblock(filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (i+1,), resblock(filters[3], filters[3], downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', resblock(filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d'%(i+1,), resblock(filters[4], filters[4], downsample=False))

        self.gap = torch.nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Linear(filters[4], outputs, bias=False)

    def forward(self, input):
        input = self.quant(input)
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.dequant(input)
        input = self.gap(input)
        input = self.quant(input)
        input = torch.flatten(input, start_dim=1)
        input = self.fc(input)
        input = self.dequant(input)
        return input

class ResNet(Resnet): 

    def __init__(self, in_channels = 1, resblock = ResBottleneckBlock, repeat = [3, 4, 6, 3], useBottleneck=True, outputs=2, seed = 12345, pretrained = False):
        super().__init__(in_channels, resblock, repeat, useBottleneck, 2)
        if pretrained:
            checkpoint = 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
            ckpt = torch.hub.load_state_dict_from_url(checkpoint, progress=False)
            new = list(ckpt.items())
            
            my_model_kvpair = self.state_dict()

            
            count=0
            for key,value in my_model_kvpair.items():
                layer_name, weights=new[count] 
                if value.shape == weights.shape:
                    my_model_kvpair[key]= weights
                count+=1
            
            self.load_state_dict(my_model_kvpair, strict=False)
        
        self.fc = torch.torch.nn.Linear(2048, outputs)
