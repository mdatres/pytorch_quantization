import torch 
import torch.nn as nn
from .resnet import ResNet
from .resnet import BasicBlock

class MyEnsemble(nn.Module):
    def __init__(self, nb_classes=2):
        super(MyEnsemble, self).__init__()
        #self.modelA = ResNet(resblock=BasicBlock)
        self.modelB = ResNet(useBottleneck= True)
        # self.modelC = ResNet(useBottleneck = True, repeat = [3, 8, 36, 3])
        #in_featA = self.modelA.fc.in_features
        in_featB = self.modelB.fc.in_features
        #in_featC = self.modelC.fc.in_features
        # self.modelA.fc = nn.Sequential(nn.Linear(in_featA,500), nn.ReLU(), nn.Dropout(), nn.Linear(500,2)) 
        self.modelB.fc = nn.Sequential(nn.Linear(in_featB,500), nn.ReLU(), nn.Dropout(), nn.Linear(500,2))
        # self.modelC.fc = nn.Sequential(nn.Linear(in_featC,500), nn.ReLU(), nn.Dropout(), nn.Linear(500,2))  
        # Remove last linear layer
        #self.modelA.fc = nn.Identity()
        #self.modelB.fc = nn.Identity()
        #self.modelC.fc = nn.Identity()
        
        # Create new classifier
        # self.classifier = nn.Linear(6, nb_classes)
        # #self.classifier = nn.Sequential(nn.Linear(4608, 500), nn.ReLU(), nn.Dropout(), nn.Linear(500,2))
        # self.fin_relu = nn.ReLU(inplace= False)

    def forward(self, x):
        # x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        # x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        # x2 = x2.view(x2.size(0), -1)
        # x3 = self.modelC(x)
        # x3 = x3.view(x3.size(0), -1)
        
        # x = torch.cat((x1, x2, x3), dim=1)
        
        # x = self.classifier(self.fin_relu(x))
        return x2