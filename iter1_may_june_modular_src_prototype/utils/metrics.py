"""
Utility Metrics
================
Fairness gaps, Cohen's d, paired t-tests, serialization helpers.
"""

from typing import Dict, List
import numpy as np


def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return float((np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-8))


def paired_test(erm_values, dro_values):
    """Paired statistical test comparing ERM vs DRO across seeds."""
    from scipy import stats
    
    erm_values = np.array(erm_values)
    dro_values = np.array(dro_values)
    
    if len(erm_values) < 3:
        return {'t_stat': float('nan'), 'p_value': float('nan'), 'cohens_d': float('nan')}
    
    t_stat, p_value = stats.ttest_rel(erm_values, dro_values)
    d = compute_cohens_d(erm_values, dro_values)
    
    return {
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(d),
    }


def compute_fairness_summary(male_mae, female_mae, race_maes):
    """Compute fairness gap metrics."""
    gender_gap = abs(male_mae - female_mae)
    race_gap = max(race_maes.values()) - min(race_maes.values()) if race_maes else 0.0
    
    return {
        'gender_gap': float(gender_gap),
        'race_gap': float(race_gap),
        'male_mae': float(male_mae),
        'female_mae': float(female_mae),
        'race_maes': {k: float(v) for k, v in race_maes.items()},
    }


def convert_to_serializable(obj):
    """Recursively convert numpy/torch types to Python-native for JSON."""
    import torch
    from pathlib import Path
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    else:
        return obj
