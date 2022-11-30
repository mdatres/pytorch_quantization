import torch 
import torch.nn as nn
from .resnet50_sicura import ResNet50_sicura
from .resnet34_sicura import ResNet34_sicura
from .resnet152_sicura import ResNet152_sicura
import torch.nn.functional as F

class MyEnsemble(nn.Module):
    def __init__(self, nb_classes=2):
        super(MyEnsemble, self).__init__()
        self.modelA = ResNet34_sicura()
        self.modelA =  self.modelA.net
        self.modelB = ResNet50_sicura()
        self.modelB =  self.modelB.net
        self.modelC = ResNet152_sicura()
        self.modelC =  self.modelC.net
        self.classifier = nn.Linear(6, nb_classes)

        

    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x3 = self.modelC(x)
        x3 = x3.view(x3.size(0), -1)

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.classifier(F.relu(x))

        return x