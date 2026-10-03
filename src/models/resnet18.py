"""
Opacus-Compatible ResNet-18 for UTKFace
=========================================
Uses GroupNorm instead of BatchNorm for DP-SGD compatibility.
"""

import torch
import torch.nn as nn
from opacus.validators import ModuleValidator


class GNBlock(nn.Module):
    """Basic residual block with GroupNorm (Opacus-compatible)."""
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, planes), planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, planes), planes)
        self.downsample = downsample
    
    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.downsample:
            x = self.downsample(x)
        return self.relu(out + x)


class OpacusResNet18(nn.Module):
    """ResNet-18 with GroupNorm for Opacus DP-SGD compatibility."""
    
    def __init__(self, num_outputs=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(min(32, 64), 64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_outputs)
    
    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.GroupNorm(min(32, planes), planes)
            )
        layers = [GNBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(GNBlock(planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.gn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1)).squeeze(-1)


def build_resnet18(num_outputs=1):
    """Build and validate an Opacus-compatible ResNet-18."""
    model = OpacusResNet18(num_outputs=num_outputs)
    if ModuleValidator.validate(model, strict=False):
        model = ModuleValidator.fix(model)
    return model
