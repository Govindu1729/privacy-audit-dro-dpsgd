# Project Worklog: DRO × DP-SGD Privacy Audit

**Project Duration:** October 2024 – November 2026  
**Status:** ✅ Complete — All experiments finalized, paper drafted

---

## Project Overview

This project investigates whether Distributionally Robust Optimization (DRO)—a fairness-promoting training method—remains safe under Differentially Private SGD (DP-SGD). We audit membership inference attack (MIA) vulnerability across three datasets (UTKFace, Texas Hospital, Adult), multiple privacy budgets (ε = 2–16), and both black-box and white-box attacks.

**Core Research Question:** Does DRO's fairness benefit come at a hidden privacy cost, and does DP-SGD fully mask it?

---

## Key Findings (Final)

### Claim Verification

| Claim | Result | Evidence |
|-------|--------|----------|
| **C1 (Fairness)** | ✅ Confirmed | DRO reduces racial disparity by **39–55%** across Adult, Texas, UTKFace |
| **C2 (Privacy Cost)** | ✅ Confirmed | DRO increases WB MIA leakage by **ΔWB = +0.053 to +0.058** (p < 10⁻³⁰) |
| **C3 (Disparate Vulnerability)** | ✅ Confirmed | Leakage concentrated in minorities (Pearson r = -0.83 to -0.91) |

### Headline Results

- **Adult Dataset:** ΔWB = +0.058, Race gap reduction = 41.8%
- **Texas Hospital:** ΔWB = +0.058, Race gap reduction = 55.1%
- **UTKFace:** ΔWB = +0.053, MAE gap reduction = 38.8%

**DP-SGD Attenuation:** The trade-off remains statistically significant at ε=8 but falls below detection threshold at ε≤4.

---

## Project Timeline & Milestones

### Phase 1: Initial Setup (Oct 2024)
- Established core audit pipeline
- Implemented Steinke one-run canary framework
- Set up DP-SGD with Opacus integration

### Phase 2: Methodological Debugging (Nov 2024 – Jan 2025)
**Major Letdowns Discovered & Fixed:**
- **FM-01:** Synthetic data fallback corrupted early results → hardened data loaders
- **FM-18:** Stale checkpoint resumption → implemented validation checks
- **FM-19:** DRO α oscillation → geometric bisection algorithm
- **FM-24/27/33:** Linux OOM killer & orphan Jupyter processes → memory management

### Phase 3: The Sampling Conflict (Feb 2025)
**Critical Discovery:** Fundamental incompatibility between DP-SGD's uniform sampling requirement and DRO's adaptive reweighting.

**Resolution:** Implemented **sub-epoch sampling** strategy—DRO samples a fraction of data each epoch while maintaining DP-SGD's uniform sampling within that subset.

### Phase 4: Notebook Bug Epidemic (Mar–Apr 2025)
**Critical Bugs Found in Notebooks 10 & 11:**
- **FM-34:** White-box sign flipped (missing `WB_NEGATE`)
- **FM-35:** Fictional ε axis (no `recalibrate_sigma`)
- **FM-36:** Dead black-box channel (unbalanced canary labels)
- **FM-37:** DRO λ saturated at 100% (variance measured under q instead of p)
- **FM-40:** Double-count bug in fairness metrics

All bugs systematically fixed across 40+ failure modes.

### Phase 5: The Fabricated PDFs Incident (May 2025)
**Critical Integrity Issue:** Two placeholder PDFs were created with invented numbers before hardened sweeps completed.

**Resolution:**
- Quarantined fabricated documents
- Implemented strict validation gates
- Re-ran all experiments with hardened notebooks
- Established automated quality rubric (C1, C2, C3 checks)

### Phase 6: Final Hardened Sweeps (Jun–Oct 2026)
- Completed full ε-sweeps (ε ∈ {∞, 8, 4, 2}) across all three datasets
- Validated all three claims with corrected methodology
- Generated publication-quality figures
- Drafted comprehensive paper

### Phase 7: Paper Finalization (Nov 2026)
- Completed LaTeX manuscript with all results
- Generated 7 publication-quality figures
- Documented all architectural decisions
- Created comprehensive appendix materials

---

## Technical Architecture

### Training Pipeline
```
Data Loading → Canary Pool Setup → Training Stages → Privacy Audit
                                      ↓              ↓
                               Phase 1: ERM      Black-Box (LiRA)
                               Phase 2: DRO      White-Box (Gradient Direction)
```

### Key Components
- **DP-SGD Layer:** GroupNorm (not BatchNorm), per-sample gradient clipping, Opacus integration
- **Hashimoto χ²-DRO:** Dual-variable updates with α auto-calibration from converged ERM runs
- **Smart Checkpoint System:** Multi-stage checkpointing, automatic resume, quality assessment
- **Auto-Tuner:** Geometric bisection for α selection, saturation detection

### Quality Rubric
| Criterion | Threshold | Purpose |
|-----------|-----------|---------|
| **C1: Fairness** | RaceGap drops ≥10% | Verify DRO improves fairness |
| **C2: Privacy Leakage** | WB AUC exceeds ERM by ≥0.02 | Measure privacy cost |
| **C3: Disparate Vulnerability** | Race disparity in AUC ≥0.03 | Check if minorities bear cost |

---

## Datasets & Models

| Dataset | Task | Model | Parameters | Privacy Budgets |
|---------|------|-------|------------|-----------------|
| **Adult** | Income prediction | DeepTabularResNet | 822k | ε ∈ {∞, 8, 4, 2} |
| **Texas Hospital** | Discharge classification | TabularMLP | 185k | ε ∈ {∞, 8, 4, 2} |
| **UTKFace** | Age regression | ResNet-18-GN | 11.2M | ε ∈ {∞, 8, 4, 2} |

---

## Key Mechanistic Insight

**The Memorization Account:** DRO's worst-case reweighting assigns higher importance to minority samples, which the model then memorizes more aggressively. This manifests as:

1. **Elevated gradient norms:** Mean ‖g‖ rises by 18–31% under DRO (paired p < 10⁻⁸)
2. **Inverse size–leakage correlation:** Pearson r between subgroup size and ΔAUC ranges from -0.83 to -0.91
3. **Dispersion widening:** Subgroup AUC spread increases from 0.010 to 0.029 under DRO

These three lines of evidence converge on a single mechanism: DRO's χ² penalty upweights high-loss minority samples, the model responds by fitting them more aggressively (larger gradients), and this aggressive fitting leaves a detectable signature that membership inference exploits.

---

## Practical Implications

1. **DRO carries a hidden privacy cost:** Practitioners deploying DRO without DP should audit for MIA vulnerability, particularly in minority subgroups.

2. **DP-SGD is necessary but not sufficient at ε ≥ 8:** At mild privacy budgets, the relative vulnerability increase persists. Tighter budgets (ε ≤ 4) are needed to fully mask the signal.

3. **Minority subgroups bear disproportionate risk:** The fairness intervention designed to protect minorities paradoxically increases their privacy exposure in the absence of strong DP.

4. **Joint design is needed:** Naive stacking of DRO inside DP-SGD leaves a measurable gap. Future work should develop privacy-aware fairness algorithms that optimize both objectives simultaneously.

---

## Repository Structure

```
privacy-audit/
├── notebooks/                    # End-to-end sweep runners
│   ├── 10_eps_sweep_adult.ipynb  # Adult ε-sweep (complete)
│   ├── 11_eps_sweep_utkface.ipynb # UTKFace ε-sweep (complete)
│   └── 09_eps_sweep_texas.ipynb  # Texas ε-sweep (complete)
│
├── paper/                        # LaTeX manuscript
│   ├── main.tex                  # Main paper
│   └── figures/                  # Publication-quality figures
│
├── results/                      # Final audit outputs
│   ├── sweep_summary.json        # Aggregated results
│   └── subgroup_tables/          # Per-dataset breakdowns
│
├── src/                          # Core implementation
│   ├── checkpoint/               # Smart checkpoint manager
│   ├── models/                   # CompactCNN, Deep ResNet, ResNet-18-GN
│   ├── trainers/                 # DRO/DP-SGD training loops
│   └── audit/                    # MIA implementations
│
├── scripts/                      # Visualization utilities
├── archive/                      # Legacy notebooks (historical reference)
├── README.md                     # Project overview
├── WORKLOG.md                    # This file
└── .gitignore
```

---

## Known Failure Modes (Resolved)

### Methodological Artifacts
| ID | Issue | Status |
|----|-------|--------|
| MA-01 | Val-sampled canary labels (BB AUC ~0.50) | ✅ Fixed: Balanced labels |
| MA-02 | Unconditional WB clipping | ✅ Fixed: Clip only when σ≠None |
| MA-03 | Static σ→ε maps | ✅ Fixed: Live recalibration |
| MA-04 | Pilot-based α calibration | ✅ Fixed: Auto-calibrate from ERM |
| MA-05 | Auditing best-val weights | ✅ Fixed: Audit final weights |
| MA-06 | WB sign convention bug | ✅ Fixed: Sign correction applied |

### Engineering Bugs
| ID | Issue | Status |
|----|-------|--------|
| FM-18 | Stale checkpoint resumption | ✅ Fixed: Validation check |
| FM-19 | DRO α oscillation | ✅ Fixed: Geometric bisection |
| FM-20 | No-op filter stripping overrides | ✅ Fixed: Filter removed |
| FM-24 | Linux OOM (System RAM) in UTKFace | ✅ Fixed: gc.collect(), batch_size 512→128, nohup launch |
| FM-25 | Papermill not in PATH under non-interactive SSH | ✅ Fixed: Use absolute path to papermill |
| FM-33 | Zombie GPU processes | ✅ Fixed: Process cleanup scripts |
| FM-34-40 | Notebook bugs (sign flip, ε axis, etc.) | ✅ Fixed: Systematic patching |

---

## Computational Resources

- **GPU Executor:** IITGN Flair2 (2× NVIDIA L40S)
- **Gateway Host:** MacBook Pro (M-series, macOS)
- **Remote Execution:** SSH bridge with papermill
- **Total Compute:** ~500 GPU-hours across all sweeps

---

## Future Work

1. **Joint Privacy-Fairness Optimization:** Develop algorithms that explicitly optimize both objectives simultaneously
2. **Extended Attack Surfaces:** Audit attribute inference and model inversion attacks
3. **Additional DRO Variants:** Test CVaR-DRO and KL-DRO privacy profiles
4. **Multi-Seed Analysis:** Add variance estimates across random seeds
5. **Theoretical Analysis:** Formalize the fairness-privacy trade-off bounds

---

## Citation

```bibtex
@article{govindu2026privacy,
  title={The Hidden Privacy Cost of Fairness: Distributionally Robust Optimization Amplifies Membership Inference in Minority Subgroups},
  author={Govindu Srimaan},
  year={2026},
  journal={Privacy Auditing Project},
  url={https://github.com/Govindu1729/privacy-audit-dro-dpsgd}
}
```

---

## Contact & Contributions

**Primary Investigator:** Govindu Srimaan  
**Repository:** https://github.com/Govindu1729/privacy-audit-dro-dpsgd  
**Documentation:** Obsidian Vault `/Projects/Privacy Auditing/`

For questions, issues, or collaboration inquiries, please open a GitHub issue or contact via the repository.

---

**Last Updated:** 2026-11-11  
**Total Project Duration:** ~2 years  
**Status:** Complete and paper-ready ✅
