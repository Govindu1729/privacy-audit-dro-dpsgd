"""
DP-SGD + DRO Trainer: Naive Stacking Implementation
=====================================================

This is the CORE module implementing the naive stacking approach:

Architecture:
┌─────────────────────────────────────────────────────────┐
│  DRO Layer (Sub-epoch Sampling)                          │
│  - Computes q_t distribution over training data          │
│  - Samples subset according to q_t each epoch            │
│  - Updates lambda_t and q_t at end of each epoch         │
├─────────────────────────────────────────────────────────┤
│  DP-SGD Layer (Opacus)                                   │
│  - Uniform sampling within each epoch                    │
│  - Per-sample gradient computation                       │
│  - Gradient clipping (max_grad_norm)                     │
│  - Gaussian noise addition                               │
│  - RDP privacy accounting                                │
└─────────────────────────────────────────────────────────┘

Key Design Decision:
- DRO affects SAMPLING (which data is seen each epoch)
- DP-SGD handles GRADIENT MECHANICS (how gradients are processed)
- This preserves Opacus's privacy accounting (uniform within epoch)
- This is "naive" because it doesn't jointly optimize both objectives

DRO Update (Hashimoto Algorithm 1, chi-squared divergence):
- lambda_{t+1} = clip(lambda_t + eta * (V_t - alpha), 0, 1/(2*alpha))
- q_{t+1} = normalize(q_t * (1 + lambda_t * (L_i - mu_t)))

NOTE: The sign convention is (V_t - alpha), NOT (alpha - V_t).
This ensures lambda increases when variance exceeds the target.
"""

from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager


# Fix Opacus BatchMemoryManager for Jupyter environments
def _patched_batch_memory_iter(self):
    for batch in self.data_loader:
        for i in range(0, len(batch[0]), self.max_physical_batch_size):
            yield tuple(x[i:i + self.max_physical_batch_size] for x in batch)

BatchMemoryManager.__iter__ = _patched_batch_memory_iter


class DPDROTrainer:
    """
    Naive Stacking: DP-SGD + DRO (Hashimoto Algorithm 1).
    
    Handles both ERM and DRO modes, with or without DP.
    """
    
    def __init__(self, config, device, model, full_train_dataset,
                 train_indices, train_transform):
        """
        Args:
            config: ExperimentConfig
            device: torch device
            model: nn.Module (unwrapped, before Opacus)
            full_train_dataset: The full raw dataset
            train_indices: List of indices used for training
            train_transform: Transform to apply during training
        """
        self.config = config
        self.device = device
        self.full_train_dataset = full_train_dataset
        self.train_indices = train_indices
        self.train_transform = train_transform
        self.n_train = len(train_indices)
        
        # DRO state
        self.q_t = np.ones(self.n_train) / self.n_train
        self.lambda_t = config.dro.lambda_init
        self.dro_enabled = config.dro.enabled
        
        # DP state
        self.dp_enabled = config.dp.enabled
        self.privacy_engine = PrivacyEngine()
        
        # Model and optimizer (set up in prepare())
        self.model = model
        self.optimizer = None
        self.criterion = None
    
    def prepare(self, train_loader):
        """
        Wrap model with Opacus if DP is enabled.
        Returns the (possibly wrapped) data loader.
        """
        self.model = self.model.to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay
        )
        
        if self.config.dataset_name == 'utkface':
            self.criterion = nn.L1Loss(reduction='mean')
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
        if self.dp_enabled:
            self.model, self.optimizer, train_loader = \
                self.privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=train_loader,
                    noise_multiplier=self.config.dp.noise_multiplier,
                    max_grad_norm=self.config.dp.max_grad_norm
                )
        
        return train_loader
    
    def get_dro_epoch_indices(self, epoch):
        """
        DRO Sub-epoch Sampling:
        Sample a subset of training data according to q_t.
        Returns None if DRO is not active for this epoch.
        """
        if not self.dro_enabled:
            return None
        if epoch <= self.config.dro.warmup_epochs:
            return None
        
        n_samples = int(self.n_train * self.config.dro.sub_epoch_fraction)
        sampled = np.random.choice(
            self.n_train, size=n_samples, p=self.q_t, replace=True
        )
        sampled = list(dict.fromkeys(sampled))  # Remove duplicates, preserve order
        return np.array(sampled)
    
    def update_dro(self, eval_loader):
        """
        End-of-epoch DRO update (Hashimoto Algorithm 1).
        
        1. Compute losses on full training set
        2. Update lambda_t (dual variable) with CORRECT sign: (V_t - alpha)
        3. Update q_t (sampling distribution)
        """
        if not self.dro_enabled:
            return
        
        self.model.eval()
        all_losses = []
        
        with torch.no_grad():
            for batch in eval_loader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                preds = self.model(inputs)
                
                if self.config.dataset_name == 'utkface':
                    losses = (preds - targets).abs()
                else:
                    losses = nn.functional.binary_cross_entropy_with_logits(
                        preds, targets, reduction='none'
                    )
                all_losses.append(losses.detach().cpu().numpy())
        
        all_losses = np.concatenate(all_losses)
        
        # Weighted mean and variance under q_t
        mu_t = np.sum(self.q_t * all_losses)
        V_t = np.sum(self.q_t * (all_losses - mu_t) ** 2)
        
        # CORRECT dual ascent update: lambda increases when V_t > alpha
        alpha = self.config.dro.alpha
        eta = self.config.dro.eta
        self.lambda_t = np.clip(
            self.lambda_t + eta * (V_t - alpha),  # (V_t - alpha), NOT (alpha - V_t)
            0.0,
            1.0 / (2.0 * alpha)
        )
        
        # Update sampling distribution
        q_new = self.q_t * (1.0 + self.lambda_t * (all_losses - mu_t))
        q_new = np.maximum(q_new, 1e-12)
        self.q_t = q_new / q_new.sum()
        
        self.model.train()
    
    def get_dro_state(self):
        """Return current DRO state for checkpointing."""
        if not self.dro_enabled:
            return {}
        return {
            'q_t': self.q_t.copy(),
            'lambda_t': float(self.lambda_t),
        }
    
    def set_dro_state(self, state):
        """Restore DRO state from checkpoint."""
        if state and self.dro_enabled:
            self.q_t = state.get('q_t', self.q_t)
            self.lambda_t = state.get('lambda_t', self.lambda_t)
    
    def get_epsilon(self):
        """Get current privacy budget spent."""
        if not self.dp_enabled:
            return float('inf')
        try:
            return self.privacy_engine.get_epsilon(self.config.dp.delta)
        except Exception:
            return 0.0
    
    def train_epoch(self, train_loader):
        """Train for one epoch. Returns average loss."""
        self.model.train()
        epoch_loss = 0.0
        epoch_n = 0
        
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=self.config.dp.physical_batch_size,
            optimizer=self.optimizer
        ) as loader_iter:
            for batch in loader_iter:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                loss = self.criterion(self.model(inputs), targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
                epoch_n += inputs.size(0)
        
        return epoch_loss / max(epoch_n, 1)
    
    def get_base_model(self):
        """Get the unwrapped model (for checkpointing)."""
        if hasattr(self.model, '_module'):
            return self.model._module
        return self.model
