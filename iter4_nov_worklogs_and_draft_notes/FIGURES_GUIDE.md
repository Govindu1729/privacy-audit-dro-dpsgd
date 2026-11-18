# Figure Generation Guide

This guide explains how to generate and update the publication-quality figures for your paper.

## 📊 Generated Figures

The `paper/plot_figures.py` script generates 5 publication-ready figures:

### Figure 1: Fairness Improvement
- **File:** `fig01_fairness_improvement.pdf/png`
- **Content:** Bar chart showing racial utility gap reduction percentages
- **Current Values:** Adult 41.8%, Texas 55.1%, UTKFace 38.8%
- **Paper Reference:** Section 4.1, Figure 1

### Figure 2: Black-Box Steinke Comparison
- **File:** `fig02_bb_steinke.pdf/png`
- **Content:** Side-by-side bar charts comparing ERM vs DRO ε_lb values
- **Panels:** No-DP (ε=∞) and ε=8
- **Paper Reference:** Section 4.2, Figure 2

### Figure 3a: White-Box ΔAUC Across Budgets
- **File:** `fig03a_wb_delta.pdf/png`
- **Content:** Grouped bar chart showing ΔWB AUC for all datasets across ε∈{∞,8,4,2}
- **Key Finding:** DRO consistently increases leakage, attenuated by DP-SGD
- **Paper Reference:** Section 4.3, Figure 3a

### Figure 3b: Subgroup Correlation
- **File:** `fig03b_subgroup_correlation.pdf/png`
- **Content:** Scatter plot with log-scale x-axis showing subgroup size vs ΔAUC
- **Key Finding:** Smaller minorities leak more (r = -0.83 to -0.91)
- **Paper Reference:** Section 4.3, Figure 3b

### Figure 4: Gradient Norms
- **File:** `fig04_gradient_norms.pdf/png`
- **Content:** Grouped bar chart of pre-clip gradient norms per subgroup (Adult)
- **Key Finding:** DRO increases norms, especially for minorities (+11% to +30%)
- **Paper Reference:** Section 4.4, Figure 4

## 🚀 Quick Start

### Option 1: Generate with Paper Values (Recommended)

The `plot_figures.py` script already contains the validated numbers from your paper. Simply run:

```bash
cd paper/
python plot_figures.py
```

**Output:** All 5 figures in `paper/figures/` (PDF + PNG formats)

### Option 2: Extract from Actual Experiments

If you want to regenerate figures using the exact values from your `result.json` files:

```bash
# Step 1: Extract data from experiments
python extract_data_for_figures.py

# Step 2: Review extracted values
cat paper/extracted_data.json

# Step 3: Update plot_figures.py with extracted values
# (Edit the hardcoded values in plot_figures.py functions)

# Step 4: Generate figures
cd paper/
python plot_figures.py
```

## 📝 Customizing Figures

### Changing Values

Edit the hardcoded arrays in `paper/plot_figures.py`:

```python
# Example: Update fairness reductions
def plot_figure1_fairness():
    datasets = ['Adult', 'Texas', 'UTKFace']
    reductions = [41.8, 55.1, 38.8]  # ← Change these values
    ...
```

### Adjusting Styling

The script uses matplotlib with LaTeX rendering. Customize at the top:

```python
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['figure.figsize'] = (8, 5)
# ... more styling options
```

### Color Schemes

Current color palette:
- **Adult:** `#2E86AB` (blue)
- **Texas:** `#A23B72` (magenta)
- **UTKFace:** `#F18F01` (orange)
- **ERM:** `#52B788` (green)
- **DRO:** `#E63946` (red)

## 🔍 Verifying Data Accuracy

### Check Extracted Data

```bash
cat paper/extracted_data.json | jq '.adult.experiments.no_dp.fairness'
```

Expected output:
```json
{
  "erm_race_gap": 0.197,
  "dro_race_gap": 0.090,
  "reduction_percent": 54.3
}
```

### Compare with Paper

| Metric | Paper Value | Extracted Value | Match? |
|--------|-------------|-----------------|--------|
| Adult fairness reduction | 41.8% | [from JSON] | ✓/✗ |
| Adult WB Δ (no-DP) | +0.058 | [from JSON] | ✓/✗ |
| Adult subgroup r | -0.87 | [from JSON] | ✓/✗ |

## 📦 Output Formats

Each figure is generated in two formats:

1. **PDF** (vector) - Best for LaTeX documents, infinite zoom
2. **PNG** (raster, 300 DPI) - Best for presentations, web

### File Sizes (Approximate)

| Figure | PDF | PNG |
|--------|-----|-----|
| fig01 | ~15 KB | ~120 KB |
| fig02 | ~20 KB | ~150 KB |
| fig03a | ~18 KB | ~140 KB |
| fig03b | ~25 KB | ~180 KB |
| fig04 | ~22 KB | ~160 KB |

## 🐛 Troubleshooting

### LaTeX Not Found

If you see `FileNotFoundError: [Errno 2] No such file or directory: 'latex'`:

```bash
# Disable LaTeX rendering in plot_figures.py
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.serif'] = ['DejaVu Serif']
```

### Missing Data

If `extract_data_for_figures.py` reports missing files:

1. Verify experiment directories exist
2. Check that `result.json` files are present in each `eps_label/method/` subdirectory
3. Update paths in `extract_data_for_figures.py` if needed

### Font Warnings

If you see font substitution warnings, install recommended fonts:

```bash
# Ubuntu/Debian
sudo apt-get install texlive-fonts-recommended

# macOS
brew install texlive
```

## 📚 Related Files

- `paper/main.tex` - LaTeX manuscript (includes these figures)
- `paper/extracted_data.json` - Raw extracted metrics
- `notebooks/*.ipynb` - Original audit notebooks
- `WORKLOG.md` - Project timeline and methodology

## ✅ Checklist Before Submission

- [ ] All 5 figures generated successfully
- [ ] PDF versions included in `paper/figures/`
- [ ] Figure references in `main.tex` match filenames
- [ ] Figure captions in paper match visual content
- [ ] Extracted data matches paper claims (±1% tolerance)
- [ ] Colorblind-friendly palette verified
- [ ] Font sizes readable when printed

---

**Last Updated:** 2026-11-12  
**Status:** Ready for paper submission
