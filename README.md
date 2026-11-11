# Privacy Audit: DRO × DP-SGD Fairness-Privacy Trade-Off

## 🎯 Project Status: **COMPLETE**

**Research Question:** Does DRO's fairness benefit come at a hidden privacy cost, and does DP-SGD fully mask it?

## 📊 Final Results

### Core Finding
Distributionally Robust Optimization (DRO) improves fairness by **38.8–55.1%** across three datasets, but simultaneously increases white-box membership inference leakage by **ΔWB = +0.053 to +0.058** in the non-private setting. The additional leakage is **disproportionately concentrated in minority subgroups** (Pearson r = -0.83 to -0.91).

### Three Claims Validated ✅

| Claim | Result | Evidence |
|-------|--------|----------|
| **C1: Fairness** | DRO reduces racial disparity by 39–55% | Adult: 41.8%, Texas: 55.1%, UTKFace: 38.8% |
| **C2: Privacy Cost** | DRO increases WB MIA leakage | ΔWB = +0.053 to +0.058 (p < 10⁻³⁰) |
| **C3: Disparate Vulnerability** | Leakage concentrated in minorities | r = -0.83 to -0.91 (subgroup size vs ΔAUC) |

### DP-SGD Attenuation

| Privacy Budget (ε) | Adult ΔWB | Texas ΔWB | UTKFace ΔWB | Statistical Significance |
|-------------------|-----------|-----------|-------------|-------------------------|
| **∞ (No DP)** | +0.058 | +0.058 | +0.053 | p < 10⁻³⁰ |
| **8** | +0.018 | +0.018 | +0.019 | p < 10⁻⁶ |
| **4** | +0.008 | +0.009 | +0.009 | p < 0.05 |
| **2** | +0.003 | +0.004 | +0.003 | p > 0.1 (not significant) |

**Key Insight:** DP-SGD reduces absolute leakage monotonically, but the **relative DRO-vs-ERM gap remains measurable at ε=8** and becomes practically masked only at ε≤4.

## 🔬 Mechanistic Analysis

### Pre-Clip Gradient Norms (Adult, Non-Private)

DRO increases mean pre-clip gradient norms, especially for smaller minority groups:

<img width="1536" height="1024" alt="Pre-clip gradient norms" src="https://github.com/user-attachments/assets/d29ea9e2-cac0-4af0-84af-3a59ef719ea1" />

| Subgroup | Sample Size | ERM ‖g‖ | DRO ‖g‖ | Increase |
|----------|-------------|---------|---------|----------|
| White | 27,816 | 10.7 | 11.9 | +11% |
| Black | 3,124 | 11.2 | 13.6 | +21% |
| Asian-Pac-Islander | 1,039 | 11.8 | 13.6 | +15% |
| Amer-Indian-Eskimo | 311 | 12.1 | 15.6 | +29% |
| Other | 271 | 12.4 | 16.1 | +30% |

**Key Observations:**

| Observation | Implication |
|---|---|
| **Magnitude Channel Annihilation** | Under DP (ε ≤ 8), ‖g‖ ≈ 27-34 while C=1.0. Clipping destroys the magnitude channel entirely. |
| **DP Attenuates but Doesn't Eliminate** | At ε=8, ΔWB remains statistically significant (p = 4.3 × 10⁻⁷). Only at ε≤4 does the signal fall below detection threshold. |
| **Disparate Impact Persists** | The "Other" subgroup on Adult experiences ΔAUC = +0.096—more than double the increase for the White majority (+0.041). |
| **Fairness-Privacy Paradox** | The fairness intervention designed to protect minorities paradoxically increases their privacy exposure in the absence of strong DP. |

**Interpretation:** Higher gradient norms indicate increased memorization of minority samples, consistent with the memorization account of privacy leakage.

## 📁 Repository Structure

```
privacy-audit-dro-dpsgd/
├── README.md                    # This file
├── WORKLOG.md                   # Complete project timeline & methodology
├── paper/
│   ├── main.tex                # LaTeX manuscript (6 pages + references)
│   ├── references.bib          # Bibliography
│   ├── figures/                # Publication-quality figures (PDF/PNG)
│   └── plot_figures.py         # Script to regenerate all figures
├── notebooks/
│   ├── 10_eps_sweep_adult.ipynb
│   ├── 11_eps_sweep_utkface.ipynb
│   └── 12_eps_sweep_texas.ipynb
└── results/
    └── [JSON audit outputs]
```

## 🚀 Quick Start

### Regenerate Publication Figures

```bash
cd paper/
python plot_figures.py
```

This generates all 5 publication-quality figures in `paper/figures/`:
- `fig01_fairness_improvement.pdf/png` - Racial utility gap reduction
- `fig02_bb_steinke.pdf/png` - Black-box Steinke ε_lb comparison
- `fig03a_wb_delta.pdf/png` - ΔWB AUC across privacy budgets
- `fig03b_subgroup_correlation.pdf/png` - Subgroup size vs ΔAUC
- `fig04_gradient_norms.pdf/png` - Pre-clip gradient norms

### Run Full Audit Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run Adult audit
jupyter notebook notebooks/10_eps_sweep_adult.ipynb

# Run UTKFace audit
jupyter notebook notebooks/11_eps_sweep_utkface.ipynb

# Run Texas audit
jupyter notebook notebooks/12_eps_sweep_texas.ipynb
```

## 🧪 Experimental Setup

### Datasets
- **Adult** (Census): 32,561 samples, income prediction, 5 racial groups
- **Texas Hospital**: 47,000+ samples, readmission prediction, 5 racial groups
- **UTKFace**: 23,708 face images, age regression, 5 racial groups

### Models
- **Adult/Texas**: TabularMLP (185k–822k parameters)
- **UTKFace**: ResNet-18 with GroupNorm (11.2M parameters)

### Privacy Budgets
- ε ∈ {∞, 8, 4, 2} with δ = 10⁻⁵
- DP-SGD with auto-calibrated noise multiplier σ

### Auditing
- **Black-box**: Loss-drop membership inference
- **White-box**: Gradient alignment membership inference
- **Steinke one-run audit** with 1000 canaries per subgroup
- **Paired Wilcoxon tests** + Cohen's d effect sizes

## 📖 Paper Summary

### Abstract
Prior work has suggested that fairness interventions may increase membership inference risk by forcing models to memorize upweighted minority samples. We test this hypothesis for χ²-distributionally robust optimization (DRO) under differentially private stochastic gradient descent (DP-SGD) using the Steinke one-run audit. Across three datasets (Adult, Texas Hospital, UTKFace), three architectures (185k–11.2M parameters), and four privacy budgets (ε ∈ {∞, 8, 4, 2}), DRO consistently reduces racial utility disparity by 38.8–55.1%, but increases white-box membership inference leakage by ΔWB between +0.053 and +0.058 in the non-private setting. Subgroup analysis shows the additional leakage is concentrated in smaller minority groups: the correlation between subgroup size and ΔAUC ranges from −0.91 to −0.83. Mechanistically, pre-clip gradient norms rise under DRO on Adult, consistent with a memorization account. DP-SGD reduces absolute leakage monotonically as ε decreases; however, the relative DRO-vs-ERM gap remains measurable at ε = 8 and becomes practically masked only at ε ≤ 4. These results reveal a measurable fairness–privacy trade-off that naive stacking of DRO and DP-SGD does not fully resolve.

### Key Contributions
1. **First systematic audit** of DRO's privacy cost across three domains and four privacy budgets
2. **Subgroup-level analysis** revealing disparate vulnerability (r = -0.83 to -0.91)
3. **Mechanistic evidence** via pre-clip gradient norm analysis
4. **Practical guidance**: DP-SGD at ε=8 preserves DRO's fairness benefit while reducing leakage, but ε≤4 destroys both fairness and privacy gains

## 📚 References

- Abadi et al. (2016). Deep Learning with Differential Privacy. CCS.
- Duchi & Namkoong (2021). Learning Models with Uniform Performance via DRO. Annals of Statistics.
- Hashimoto et al. (2018). Fairness Without Demographics in Repeated Loss Minimization. ICML.
- Steinke, Nasr & Jagielski (2023). Privacy Auditing with One Training Run. NeurIPS.

## 👥 Authors

**Anonymous Submission** (Double-blind review)

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Project Complete** ✅ | **Paper Ready for Submission** 📄 | **All Claims Validated** 🔬
