"""
Tabular Models for Texas Hospital and Adult Datasets
=====================================================
Logistic Regression and MLP options, Opacus-compatible.
"""

import torch
import torch.nn as nn
from opacus.validators import ModuleValidator


class LogisticRegression(nn.Module):
    """Simple logistic regression for tabular data."""
    
    def __init__(self, input_dim, num_outputs=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_outputs)
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)


class MLP(nn.Module):
    """Multi-layer perceptron for tabular data."""
    
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


def build_tabular_model(input_dim, model_type='logreg', num_outputs=1):
    """Build and validate an Opacus-compatible tabular model."""
    if model_type == 'logreg':
        model = LogisticRegression(input_dim, num_outputs)
    elif model_type == 'mlp':
        model = MLP(input_dim, num_outputs=num_outputs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if ModuleValidator.validate(model, strict=False):
        model = ModuleValidator.fix(model)
    return model
