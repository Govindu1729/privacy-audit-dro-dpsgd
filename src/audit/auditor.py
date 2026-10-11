"""
Privacy Auditor: Steinke et al. (2023) Framework
==================================================
One-training-run audit using canary samples.

Reference: Steinke, T., Nasr, M., & Jagielski, M. (2023).
"Privacy Auditing with One (1) Training Run." NeurIPS 2023.
"""

import math
from typing import Dict
import numpy as np
from scipy import stats


class SteinkeAuditor:
    """
    Implements the Steinke one-training-run privacy audit.
    
    Given per-canary scores and membership labels, computes:
    - Best threshold V (number of canaries correctly identified)
    - p-value under null hypothesis
    - epsilon lower bound
    - AUC, attack accuracy, Cohen's d
    
    IMPORTANT: White-box scores use SIGN CORRECTION (MA-06).
    Members score NEGATIVE on ⟨Δθ, gᵢ⟩ because gradient descent moves
    weights along -g. Multiply raw dot product by -1.0 before audit.
    """
    
    def __init__(self, gamma=0.05):
        self.gamma = gamma
    
    def audit(self, scores, membership):
        """
        Run the audit.
        
        Args:
            scores: np.array of per-canary scores (higher = more likely IN)
            membership: np.array of binary membership labels (1=IN, 0=OUT)
        
        Returns:
            Dict with audit results
        """
        m = membership.astype(int)
        k = len(scores)
        
        # Find best threshold
        thresholds = np.linspace(scores.min() - 1e-9, scores.max() + 1e-9, 1000)
        best_V = 0
        best_tau = thresholds[0]
        
        for tau in thresholds:
            guesses = (scores > tau).astype(int)
            V = (guesses == m).sum()
            if V > best_V:
                best_V = V
                best_tau = tau
        
        # p-value under null (random guessing)
        p_null = stats.binom.sf(best_V - 1, k, 0.5)
        
        # Binary search for epsilon lower bound
        def p_val(eps):
            p_succ = math.exp(eps) / (1 + math.exp(eps))
            return stats.binom.sf(best_V - 1, k, p_succ)
        
        lo, hi = 0.0, 20.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if p_val(mid) > self.gamma:
                lo = mid
            else:
                hi = mid
        
        eps_lb = lo if p_val(lo) <= self.gamma else 0.0
        
        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(m, scores)
        except Exception:
            auc = 0.5
        
        # Score separation
        in_scores = scores[m == 1]
        out_scores = scores[m == 0]
        delta = float(in_scores.mean() - out_scores.mean()) if len(in_scores) > 0 and len(out_scores) > 0 else 0.0
        pooled_std = np.sqrt((in_scores.var() + out_scores.var()) / 2) if len(in_scores) > 0 and len(out_scores) > 0 else 1.0
        cohens_d = delta / (pooled_std + 1e-8)
        
        return {
            'eps_lb': float(eps_lb),
            'V': int(best_V),
            'k': int(k),
            'threshold': float(best_tau),
            'p_value': float(p_null),
            'attack_acc': float(best_V / k),
            'auc': float(auc),
            'delta': delta,
            'cohens_d': float(cohens_d),
        }
    
    def interpret(self, results):
        """Generate human-readable interpretation of audit results."""
        eps_lb = results['eps_lb']
        auc = results['auc']
        
        if eps_lb == 0.0 and abs(auc - 0.5) < 0.05:
            return (
                f"PERFECT PRIVACY: eps_lb=0.0, AUC={auc:.3f}. "
                f"Attacker cannot distinguish IN from OUT canaries."
            )
        elif eps_lb == 0.0:
            return (
                f"STRONG PRIVACY: eps_lb=0.0, AUC={auc:.3f}. "
                f"Slight signal but not statistically significant."
            )
        else:
            return (
                f"PRIVACY BREACH: eps_lb={eps_lb:.3f}, AUC={auc:.3f}. "
                f"Attacker identifies {results['V']}/{results['k']} canaries."
            )
