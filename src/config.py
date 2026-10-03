"""
Central Experiment Configuration
================================
All hyperparameters in one place. Change here, everything updates.
Supports epsilon sweep via noise_multiplier parameterization.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import math


@dataclass
class DPConfig:
    """Differential Privacy configuration."""
    enabled: bool = True
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    physical_batch_size: int = 32


@dataclass
class DROConfig:
    """
    Distributionally Robust Optimization configuration.
    Implements Hashimoto et al. (2018) Algorithm 1: chi-squared divergence DRO.
    
    The dual update for lambda uses the CORRECT sign convention:
        lambda_{t+1} = clip(lambda_t + eta * (V_t - alpha), 0, 1/(2*alpha))
    
    where V_t is the weighted variance of losses under q_t.
    """
    enabled: bool = True
    mode: str = 'alg1_chisq'
    alpha: float = 50.0            # chi-squared ball radius (target variance)
    eta: float = 0.01              # step size for lambda dual update
    lambda_init: float = 0.001     # warm start for lambda
    warmup_epochs: int = 2         # epochs of pure ERM before DRO activates
    sub_epoch_fraction: float = 0.8  # fraction of data sampled per DRO epoch


@dataclass
class AuditConfig:
    """Privacy audit configuration (Steinke et al., 2023)."""
    k: int = 500                   # number of canaries
    gamma: float = 0.05            # significance level for hypothesis test
    seed: int = 42
    wb_score_epochs: int = 5       # compute WB scores every N epochs
    wb_sample_batches: int = 10    # batches to sample for WB scoring


@dataclass
class TrainConfig:
    """General training configuration."""
    epochs: int = 25
    lr: float = 1e-4
    batch_size: int = 64
    weight_decay: float = 1e-4
    val_frac: float = 0.15
    seed: int = 2024
    num_workers: int = 4
    checkpoint_every_n_epochs: int = 5


@dataclass
class ExperimentConfig:
    """Master configuration combining all sub-configs."""
    experiment_name: str = "experiment"
    dataset_name: str = "utkface"
    data_dir: Path = Path("data")
    exp_dir: Path = Path("experiments")
    
    train: TrainConfig = field(default_factory=TrainConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    dro: DROConfig = field(default_factory=DROConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    
    subgroup_columns: List[str] = field(default_factory=lambda: ['race', 'gender'])
    
    def get_output_dir(self) -> Path:
        """Get output directory, parameterized by DP config for epsilon sweep."""
        if self.dp.enabled:
            dp_tag = f"dp_sigma{self.dp.noise_multiplier}"
        else:
            dp_tag = "no_dp"
        
        dro_tag = "dro" if self.dro.enabled else "erm"
        run_name = f"{self.dataset_name}_{dro_tag}_{dp_tag}"
        
        out = self.exp_dir / self.experiment_name / run_name
        out.mkdir(parents=True, exist_ok=True)
        return out
    
    def to_dict(self) -> dict:
        """Serialize config for hashing and JSON export."""
        return {
            'experiment_name': self.experiment_name,
            'dataset_name': self.dataset_name,
            'epochs': self.train.epochs,
            'lr': self.train.lr,
            'batch_size': self.train.batch_size,
            'weight_decay': self.train.weight_decay,
            'val_frac': self.train.val_frac,
            'seed': self.train.seed,
            'dp_enabled': self.dp.enabled,
            'dp_noise_multiplier': self.dp.noise_multiplier,
            'dp_max_grad_norm': self.dp.max_grad_norm,
            'dp_delta': self.dp.delta,
            'dro_enabled': self.dro.enabled,
            'dro_mode': self.dro.mode,
            'dro_alpha': self.dro.alpha,
            'dro_eta': self.dro.eta,
            'dro_lambda_init': self.dro.lambda_init,
            'dro_warmup_epochs': self.dro.warmup_epochs,
            'audit_k': self.audit.k,
            'audit_gamma': self.audit.gamma,
        }


# ── Pre-built configurations ───────────────────────────────────────────────

def get_utkface_config(noise_multiplier: float = 1.1, dp_enabled: bool = True,
                       dro_enabled: bool = True) -> ExperimentConfig:
    """Configuration for UTKFace (vision, ResNet-18, age regression)."""
    return ExperimentConfig(
        experiment_name="utkface",
        dataset_name="utkface",
        data_dir=Path("data/UTKFace"),
        exp_dir=Path("experiments"),
        train=TrainConfig(epochs=25, lr=1e-4, batch_size=64, seed=2024),
        dp=DPConfig(enabled=dp_enabled, noise_multiplier=noise_multiplier,
                    max_grad_norm=1.0, delta=1e-5, physical_batch_size=32),
        dro=DROConfig(enabled=dro_enabled, alpha=50.0, eta=0.01,
                      lambda_init=0.001, warmup_epochs=2),
        audit=AuditConfig(k=500, gamma=0.05, seed=42),
        subgroup_columns=['race', 'gender'],
    )


def get_texas_config(noise_multiplier: float = 1.0, dp_enabled: bool = True,
                     dro_enabled: bool = True) -> ExperimentConfig:
    """Configuration for Texas Hospital (tabular, binary classification)."""
    return ExperimentConfig(
        experiment_name="texas",
        dataset_name="texas",
        data_dir=Path("data"),
        exp_dir=Path("experiments"),
        train=TrainConfig(epochs=50, lr=1e-3, batch_size=128, seed=2024),
        dp=DPConfig(enabled=dp_enabled, noise_multiplier=noise_multiplier,
                    max_grad_norm=1.0, delta=1e-5, physical_batch_size=64),
        dro=DROConfig(enabled=dro_enabled, alpha=5.0, eta=0.01,
                      lambda_init=0.001, warmup_epochs=5),
        audit=AuditConfig(k=200, gamma=0.05, seed=42),
        subgroup_columns=['race', 'gender'],
    )


def get_adult_config(noise_multiplier: float = 1.0, dp_enabled: bool = True,
                     dro_enabled: bool = True) -> ExperimentConfig:
    """Configuration for Adult / Census Income (tabular, binary classification)."""
    return ExperimentConfig(
        experiment_name="adult",
        dataset_name="adult",
        data_dir=Path("data"),
        exp_dir=Path("experiments"),
        train=TrainConfig(epochs=50, lr=1e-3, batch_size=128, seed=2024),
        dp=DPConfig(enabled=dp_enabled, noise_multiplier=noise_multiplier,
                    max_grad_norm=1.0, delta=1e-5, physical_batch_size=64),
        dro=DROConfig(enabled=dro_enabled, alpha=5.0, eta=0.01,
                      lambda_init=0.001, warmup_epochs=5),
        audit=AuditConfig(k=200, gamma=0.05, seed=42),
        subgroup_columns=['race', 'gender'],
    )


def get_epsilon_sweep_configs(dataset: str = "utkface"):
    """
    Generate configs for the full epsilon sweep: epsilon in {2, 4, 8, 16}.
    Also includes a no-DP baseline.
    
    Returns list of (label, config) tuples.
    """
    getter = {
        'utkface': get_utkface_config,
        'texas': get_texas_config,
        'adult': get_adult_config,
    }[dataset]
    
    configs = []
    
    # No-DP baselines
    configs.append(("no_dp_erm", getter(dp_enabled=False, dro_enabled=False)))
    configs.append(("no_dp_dro", getter(dp_enabled=False, dro_enabled=True)))
    
    # DP-SGD with different noise levels (approximate epsilon mapping)
    # Higher noise_multiplier -> lower epsilon -> stronger privacy
    # These are approximate; actual epsilon depends on epochs, batch_size, dataset size
    noise_levels = [0.5, 1.0, 2.0, 4.0]  # Approximate epsilon: 16, 8, 4, 2
    
    for sigma in noise_levels:
        configs.append((f"dp_sigma{sigma}_erm", getter(noise_multiplier=sigma, dro_enabled=False)))
        configs.append((f"dp_sigma{sigma}_dro", getter(noise_multiplier=sigma, dro_enabled=True)))
    
    return configs
