import torch
from torch import nn
from torchvision.models import ResNet

class ResNetEncoder(nn.Module):
    def __init__(self, resnet: ResNet):
        super().__init__()
        self.resnet = resnet
        
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class FeatureCPResNet(nn.Module):
    def __init__(self, resnet: ResNet):
        super().__init__()
        self.encoder = ResNetEncoder(resnet)
        self.g = resnet.fc

    def forward(self, x):
        x = self.encoder(x)
        x = self.g(x)
        return x