"""
Subgroup Vulnerability Analyzer
================================
Computes per-subgroup MIA AUC to detect disparate vulnerability.

Reference: Kulynych et al., "Disparate Vulnerability to Membership Inference Attacks."
"""

from typing import Dict, List
import numpy as np


class SubgroupAnalyzer:
    """Analyzes MIA vulnerability per demographic subgroup."""
    
    def __init__(self, subgroup_names):
        """
        Args:
            subgroup_names: Mapping from subgroup column to list of group names
                           e.g., {'race': ['White','Black','Asian','Indian','Others'],
                                  'gender': ['Male','Female']}
        """
        self.subgroup_names = subgroup_names
    
    def analyze(self, scores, membership, subgroup_labels, min_samples=10):
        """
        Compute per-subgroup AUC.
        
        Args:
            scores: Per-sample MIA scores
            membership: Binary membership labels
            subgroup_labels: {'race': np.array([...]), 'gender': np.array([...])}
            min_samples: Skip groups with fewer samples
        
        Returns:
            Nested dict: {'race': {'White': {'auc': 0.55, ...}, ...}}
        """
        from sklearn.metrics import roc_auc_score
        
        results = {}
        
        for sg_name, labels in subgroup_labels.items():
            results[sg_name] = {}
            unique_groups = np.unique(labels)
            
            for g in unique_groups:
                mask = labels == g
                g_scores = scores[mask]
                g_membership = membership[mask]
                
                n_in = int(g_membership.sum())
                n_out = int(len(g_membership) - n_in)
                
                if n_in < min_samples or n_out < min_samples:
                    continue
                
                try:
                    auc = roc_auc_score(g_membership.astype(int), g_scores)
                except Exception:
                    auc = 0.5
                
                names = self.subgroup_names.get(sg_name, [])
                g_name = names[int(g)] if int(g) < len(names) else f"group_{g}"
                
                results[sg_name][g_name] = {
                    'auc': float(auc),
                    'n_in': n_in,
                    'n_out': n_out,
                }
        
        return results
    
    def compute_disparity(self, subgroup_results):
        """Compute disparity metrics from subgroup results."""
        disparities = {}
        
        for sg_name, groups in subgroup_results.items():
            aucs = {g: v['auc'] for g, v in groups.items()}
            if len(aucs) < 2:
                continue
            
            max_auc = max(aucs.values())
            min_auc = min(aucs.values())
            
            disparities[sg_name] = {
                'max_auc': max_auc,
                'min_auc': min_auc,
                'gap': max_auc - min_auc,
                'most_vulnerable': max(aucs, key=aucs.get),
                'least_vulnerable': min(aucs, key=aucs.get),
            }
        
        return disparities
