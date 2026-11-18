"""
Data Extraction Helper for Figure Generation
=============================================
Extracts validated results from JSON audit outputs to regenerate figures
with actual experimental data.

Usage:
    python extract_data_for_figures.py

This script reads the result.json files from completed audits and extracts
the key metrics needed for each figure:
- Figure 1: Fairness improvements (race_gap_erm, race_gap_dro)
- Figure 2: Black-box Steinke ε_lb values
- Figure 3a: White-box AUC deltas across privacy budgets
- Figure 3b: Subgroup-level ΔAUC and subgroup sizes
- Figure 4: Pre-clip gradient norms per subgroup

Output:
    paper/extracted_data.json - Structured data ready for plotting
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def load_result_json(path: Path) -> Dict[str, Any]:
    """Load a result.json file."""
    with open(path, 'r') as f:
        return json.load(f)


def extract_fairness_metrics(result: Dict) -> Dict[str, float]:
    """Extract race_gap metrics for Figure 1."""
    return {
        'race_gap_erm': result['final_metrics']['race_gap'],
        'race_gap_dro': result['final_metrics']['race_gap'],
        'race_gap_reduction': (
            (result['final_metrics']['race_gap'] - 
             result['final_metrics']['race_gap']) /
            result['final_metrics']['race_gap']
            if result['final_metrics']['race_gap'] > 0 else 0.0
        )
    }


def extract_steinke_metrics(result: Dict) -> Dict[str, float]:
    """Extract Steinke audit metrics for Figure 2."""
    return {
        'bb_eps_lb': result['bb_audit']['eps_lb'],
        'wb_eps_lb': result['wb_audit']['eps_lb'],
        'bb_auc': result['bb_audit']['auc'],
        'wb_auc': result['wb_audit']['auc'],
    }


def extract_subgroup_metrics(result: Dict) -> Dict[str, Dict]:
    """Extract subgroup-level metrics for Figures 3b and 4."""
    subgroups = {}
    
    for axis in ['race', 'gender']:
        if axis in result['subgroup_auc']:
            for group_name, group_data in result['subgroup_auc'][axis].items():
                if not group_name.startswith('_'):
                    subgroups[group_name] = {
                        'n_in': group_data['n_in'],
                        'n_out': group_data['n_out'],
                        'auc': group_data['auc'],
                        'preclip_grad_norm': group_data['preclip_grad_norm'],
                        'preclip_grad_norm_std': group_data.get('preclip_grad_norm_std', 0.0),
                    }
    
    return subgroups


def extract_all_results(experiment_root: Path, dataset: str) -> Dict[str, Any]:
    """
    Extract all metrics from a completed experiment.
    
    Args:
        experiment_root: Path to experiment directory (e.g., /path/to/adult_eps_sweep)
        dataset: Dataset name ('adult', 'texas', 'utkface')
    
    Returns:
        Structured dictionary with all extracted metrics
    """
    
    epsilon_sweep = ['no_dp', 'eps8', 'eps4', 'eps2']
    
    extracted = {
        'dataset': dataset,
        'experiments': {}
    }
    
    for eps_label in epsilon_sweep:
        eps_dir = experiment_root / eps_label
        
        # Load ERM result
        erm_path = eps_dir / 'erm' / 'result.json'
        dro_path = eps_dir / 'dro' / 'result.json'
        
        if not erm_path.exists() or not dro_path.exists():
            print(f"⚠️  Skipping {eps_label}: missing result.json files")
            continue
        
        erm_result = load_result_json(erm_path)
        dro_result = load_result_json(dro_path)
        
        # Extract paired metrics
        erm_fairness = extract_fairness_metrics(erm_result)
        dro_fairness = extract_fairness_metrics(dro_result)
        
        erm_steinke = extract_steinke_metrics(erm_result)
        dro_steinke = extract_steinke_metrics(dro_result)
        
        erm_subgroups = extract_subgroup_metrics(erm_result)
        dro_subgroups = extract_subgroup_metrics(dro_result)
        
        # Calculate deltas
        wb_delta = dro_steinke['wb_auc'] - erm_steinke['wb_auc']
        bb_delta = dro_steinke['bb_auc'] - erm_steinke['bb_auc']
        
        # Calculate fairness improvement
        if erm_fairness['race_gap_erm'] > 0:
            fairness_reduction = (
                (erm_fairness['race_gap_erm'] - dro_fairness['race_gap_dro']) /
                erm_fairness['race_gap_erm']
            )
        else:
            fairness_reduction = 0.0
        
        # Calculate subgroup-level deltas
        subgroup_deltas = {}
        for group_name in erm_subgroups.keys():
            if group_name in dro_subgroups:
                subgroup_deltas[group_name] = {
                    'n_total': erm_subgroups[group_name]['n_in'] + erm_subgroups[group_name]['n_out'],
                    'delta_auc': dro_subgroups[group_name]['auc'] - erm_subgroups[group_name]['auc'],
                    'erm_grad_norm': erm_subgroups[group_name]['preclip_grad_norm'],
                    'dro_grad_norm': dro_subgroups[group_name]['preclip_grad_norm'],
                    'grad_norm_increase': (
                        (dro_subgroups[group_name]['preclip_grad_norm'] - 
                         erm_subgroups[group_name]['preclip_grad_norm']) /
                        erm_subgroups[group_name]['preclip_grad_norm']
                        if erm_subgroups[group_name]['preclip_grad_norm'] > 0 else 0.0
                    )
                }
        
        extracted['experiments'][eps_label] = {
            'epsilon': eps_label,
            'fairness': {
                'erm_race_gap': erm_fairness['race_gap_erm'],
                'dro_race_gap': dro_fairness['race_gap_dro'],
                'reduction_percent': fairness_reduction * 100,
            },
            'steinke': {
                'bb_eps_lb_erm': erm_steinke['bb_eps_lb'],
                'bb_eps_lb_dro': dro_steinke['bb_eps_lb'],
                'wb_eps_lb_erm': erm_steinke['wb_eps_lb'],
                'wb_eps_lb_dro': dro_steinke['wb_eps_lb'],
                'bb_auc_erm': erm_steinke['bb_auc'],
                'bb_auc_dro': dro_steinke['bb_auc'],
                'wb_auc_erm': erm_steinke['wb_auc'],
                'wb_auc_dro': dro_steinke['wb_auc'],
                'bb_delta': bb_delta,
                'wb_delta': wb_delta,
            },
            'subgroups': subgroup_deltas,
            'validation': {
                'erm_validated': erm_result.get('validated', True),
                'dro_validated': dro_result.get('validated', False),
                'dro_activation_ratio': dro_result.get('dro_state_final', {}).get('activation_ratio', 0.0),
            }
        }
    
    return extracted


def main():
    """Extract data from all three datasets."""
    
    print("=" * 70)
    print("EXTRACTING EXPERIMENTAL DATA FOR FIGURE GENERATION")
    print("=" * 70)
    print()
    
    # Define experiment directories
    # Adjust these paths to match your actual experiment locations
    experiment_roots = {
        'adult': Path('/home/govindu.srimaan/project/monolith/experiments/adult_eps_sweep'),
        'texas': Path('/home/govindu.srimaan/project/monolith/experiments/texas_eps_sweep'),
        'utkface': Path('/home/govindu.srimaan/project/monolith/experiments/utkface_eps_sweep'),
    }
    
    all_extracted = {}
    
    for dataset, exp_root in experiment_roots.items():
        if not exp_root.exists():
            print(f"⚠️  {dataset}: Experiment directory not found at {exp_root}")
            continue
        
        print(f"\n📊 Extracting {dataset.upper()} data...")
        extracted = extract_all_results(exp_root, dataset)
        
        # Summary statistics
        print(f"  ✓ Extracted {len(extracted['experiments'])} privacy budget configurations")
        
        for eps_label, exp_data in extracted['experiments'].items():
            fairness = exp_data['fairness']
            steinke = exp_data['steinke']
            print(f"    {eps_label:8s}: Fairness ↓ {fairness['reduction_percent']:5.1f}%, "
                  f"WB Δ {steinke['wb_delta']:+.3f}, BB Δ {steinke['bb_delta']:+.3f}")
        
        all_extracted[dataset] = extracted
    
    # Save extracted data
    output_path = Path('paper/extracted_data.json')
    with open(output_path, 'w') as f:
        json.dump(all_extracted, f, indent=2)
    
    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\n💾 Saved extracted data to: {output_path.absolute()}")
    print()
    print("Next steps:")
    print("  1. Review paper/extracted_data.json to verify extracted metrics")
    print("  2. Update paper/plot_figures.py with actual values from extracted_data.json")
    print("  3. Run: python paper/plot_figures.py")
    print()


if __name__ == '__main__':
    main()
