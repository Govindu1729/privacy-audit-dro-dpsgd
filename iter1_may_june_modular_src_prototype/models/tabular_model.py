"""
Tabular Models for Texas Hospital and Adult Datasets
=====================================================
Deep ResNet (6 layers, 256 hidden units) with GroupNorm — Opacus-compatible.
MLP and Logistic Regression kept for backward compatibility (superseded).
"""

import torch
import torch.nn as nn
from opacus.validators import ModuleValidator


class LogisticRegression(nn.Module):
    """Simple logistic regression for tabular data (superseded)."""
    
    def __init__(self, input_dim, num_outputs=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_outputs)
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)


class MLP(nn.Module):
    """Multi-layer perceptron for tabular data (superseded by DeepResNet)."""
    
    def __init__(self, input_dim, hidden_dims=None, num_outputs=1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_outputs))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class DeepResNet(nn.Module):
    """
    Deep ResNet for tabular datasets (Texas, Adult).
    6 layers with GroupNorm for Opacus DP-SGD compatibility.
    6 layers, 256 hidden units per layer — Opacus-compatible with GroupNorm.
    """
    
    def __init__(self, input_dim, num_outputs=1, num_layers=6, hidden_dim=256):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.ReLU(inplace=False)
        )
        
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GroupNorm(min(32, hidden_dim), hidden_dim))
            layers.append(nn.ReLU(inplace=False))
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, num_outputs)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        return self.output_layer(x).squeeze(-1)


def build_deep_resnet(input_dim, num_outputs=1, num_layers=6, hidden_dim=256):
    """Build and validate Opacus-compatible Deep ResNet for tabular."""
    model = DeepResNet(input_dim, num_outputs, num_layers, hidden_dim)
    if ModuleValidator.validate(model, strict=False):
        model = ModuleValidator.fix(model)
    return model


def build_tabular_model(input_dim, model_type='deep_resnet', num_outputs=1):
    """Build and validate an Opacus-compatible tabular model.
    
    Args:
        input_dim: Input feature dimension
        model_type: 'deep_resnet' (recommended), 'mlp', or 'logreg'
        num_outputs: Number of output units
    """
    if model_type == 'deep_resnet':
        return build_deep_resnet(input_dim, num_outputs=num_outputs)
    elif model_type == 'mlp':
        model = MLP(input_dim, num_outputs=num_outputs)
    elif model_type == 'logreg':
        model = LogisticRegression(input_dim, num_outputs=num_outputs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if ModuleValidator.validate(model, strict=False):
        model = ModuleValidator.fix(model)
    return model