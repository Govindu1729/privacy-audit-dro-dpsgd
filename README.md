# Privacy Auditing Pipeline: DRO vs. DP-SGD

> [!note] **Research Status** (Updated 2026-11-08)
> This project documents systematic investigation into fairness-privacy trade-offs. Multiple methodological artifacts and engineering bugs have been identified and resolved (see `Known Failure Modes` section). Current findings include null results on tabular data due to DP masking effects. UTKFace sweep with CompactCNN in progress (2026-10-18→2026-11-08).

## 🔍 Core Research Question
Does DRO's fairness benefit come at a hidden privacy cost (Membership Inference Attack vulnerability), and does DP-SGD fully mask it? Specifically, we investigate if DP-SGD's gradient clipping erases the magnitude-channel signal that DRO amplifies in minority subgroups.

## 🏗️ Pipeline Architecture

### Training Pipeline
1. **Smart Decision Engine**: Determines fresh start vs. resume based on checkpoint hash verification
2. **Data Preparation**: Canary pool setup (500-2000 records, balanced 50/50 IN/OUT split)
3. **Training Stages**:
   - Phase 1: ERM (baseline, plain training)
   - Phase 2: Hashimoto χ²-DRO (fairness optimization)
4. **Privacy Audit**:
   - Black-Box (LiRA): Uses initial_loss − final_loss scores
   - White-Box (Gradient Direction): Uses $\langle \Delta	heta, g_i \nangle$ dot product (sign-corrected)
5. **Quality Rubric**:
   - C1: DRO improves fairness (RaceGap drops ≥10%)
   - C2: DRO leaks more privacy (WB AUC exceeds ERM by ≥0.02)
   - C3: Uneven leakage across subgroups (Race disparity in AUC ≥0.03)

### Key Components
- **DP-SGD Layer**: GroupNorm (not BatchNorm), per-sample gradient clipping, Opacus integration
- **Hashimoto DRO**: Dual-variable updates with α auto-calibration from converged ERM runs
- **Smart Checkpoint System**: Multi-stage checkpointing, automatic resume, quality assessment
- **Auto-Tuner**: Geometric bisection for α selection, saturation detection

## 📂 Repository Structure

```diff
privacy-audit/
├── notebooks/                 # End-to-end sweep runners
│   ├── 11_eps_sweep_utkface.ipynb    # UTKFace ε-sweep (in progress, flair2)
│   ├── 10_eps_sweep_adult.ipynb      # Adult ε-sweep (complete, Deep ResNet)
│   ├── 09_eps_sweep_texas.ipynb      # Texas ε-sweep (needs re-run, old notebook)
│   └── archive/               # Legacy notebooks (historical reference)
│
├── src/
│   ├── checkpoint/            # Smart checkpoint manager (10KB)
│   ├── models/                # CompactCNN (UTKFace), Deep ResNet (tabular)
│   ├── trainers/              # DRO/DP-SGD training loops
│   └── audit/                 # MIA implementations
│
├── reports/                   # Generated audit outputs
├── scripts/                   # Hyperbola visualization utilities
└── archive/                   # Legacy notebooks (historical reference)
```

## 🔬 Scientific Observations (Not Bugs)

| Observation | Implication |
|---|---|
| **Magnitude Channel Annihilation** | Under DP (ε ≤ 8), ‖g‖ ≈ 27-34 while C=1.0. Clipping destroys the magnitude channel entirely. |
| **DP Masks DRO Leakage** | On Adult, DRO increases BB leakage by +0.059 without DP. Under DP, BB AUC drops to ~0.50 for both ERM and DRO. |
| **Disparate Impact Under DP** | WB AUC near-random overall, but DRO increases vulnerability for minorities while decreasing for majorities. |
| **Utility Collapse Masks Trade-off** | Aggressive DRO tuning for fairness causes test error spike (35% vs ERM 16%). DRO BB AUC drops below ERM (0.601 vs 0.684). |

## 🐛 Known Failure Modes (Purged)

### Methodological Artifacts
| ID | Issue | Status |
|---|---|---|
| MA-01 | Val-sampled canary labels (BB AUC ~0.50) | Fixed: Balanced labels |
| MA-02 | Unconditional WB clipping | Fixed: Clip only when σ≠None |
| MA-03 | Static σ→ε maps | Fixed: Live recalibration |
| MA-04 | Pilot-based α calibration | Fixed: Auto-calibrate from ERM |
| MA-05 | Auditing best-val weights | Fixed: Audit final weights |
| MA-06 | WB sign convention bug | Fixed: Sign correction applied |

### Engineering Bugs
| ID | Issue | Status |
|---|---|---|
| FM-18 | Stale checkpoint resumption | Fixed: Validation check |
| FM-19 | DRO α oscillation | Fixed: Geometric bisection |
| FM-20 | No-op filter stripping overrides | Fixed: Filter removed |
| FM-24 | Linux OOM (System RAM) in UTKFace | Fixed: gc.collect(), batch_size 512→128, nohup launch |
| FM-25 | Papermill not in PATH under non-interactive SSH | Fixed: Use absolute path to papermill |

## ⚠️ Important Notes

- **White-Box Scores**: Members score negative on $\langle \Delta	heta, g_i \nangle$ (gradient descent moves weights along $-g$). Sign correction applied.
- **Black-Box on Adult**: Dead signal (~0.50 AUC) due to label imbalance (76% negative). Canary labels now forced 50/50.
- **DRO Convergence**: α must be auto-calibrated from the converged ERM run at the same σ. Static calibration leads to λ saturation/collapse.
- **Model Capacity**: Tabular MLPs insufficient for memorization. Deep ResNet required for MIA signal.
- **CompactCNN (UTKFace)**: 961,857 params, 128×128 images, GroupNorm — 3.4× better SNR per step vs ResNet-18
- **Deep ResNet (Texas/Adult)**: 6 layers, 256 hidden units — sufficient capacity for MIA signal

## 📊 Realized Privacy Budgets

| Dataset | σ | ERM ε | DRO ε |
|---|---|---|---|
| UTKFace (40 ep, CompactCNN) | 0.6129 | 8.0091 | TBD (in progress) |
| Texas (50 ep, Deep ResNet) | 1.0 | 8.30 | 6.26 |
| Adult (50 ep, Deep ResNet) | 1.0 | 5.44 | 4.13 |

> **Note:** DRO spends less ε per epoch due to sub-epoch sampling (0.8× data). Budget parity approximate, not exact.

## 📖 Project Documentation

Detailed methodology and decision logs are maintained in the Obsidian Vault (`/Users/srimaan/Documents/Obsidian Vault/Projects/Privacy Auditing/`):
- **Method Notes**: DP-SGD, Hashimoto χ²-DRO, The Sampling Conflict
- **Decisions**: Known Failure Modes, Hyperparameter Rationale
- **Schemas**: Config, Dataset, Canary, Checkpoint, Results JSON

## 📌 Citation

> *Privacy Audit Pipeline: DRO vs. DP-SGD* (2024-2026)
> Govindu Srimaan

**Repository**: `Govindu1729/privacy-audit`

