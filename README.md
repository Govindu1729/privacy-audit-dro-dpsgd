# The Hidden Privacy Cost of Fairness
### Auditing Distributionally Robust Optimization and Differentially Private SGD Across Demographic Subgroups

**Author:** Srimaan Govindu  
**Advisor:** Prof. Manisha Padala  
**Affiliation:** Department of Computer Science and Engineering, IIT Gandhinagar  

---

## 🌟 Start Here: Final Validated Deliverables (`final_results/`)

This repository documents a multi-month empirical and theoretical investigation (**May–June & September–November**) into how algorithmic fairness interventions interact with membership inference vulnerability and differential privacy.

While the repository preserves all intermediate prototypes, negative results, and failed hypotheses from earlier iterations (`iter1_*` through `iter4_*`), **the final, fully validated research artifacts are consolidated in [`final_results/`](./final_results/):**

| Deliverable | File in [`final_results/`](./final_results/) | Description |
| :--- | :--- | :--- |
| **1. Project Report (15 pp.)** | [`Project_Report.pdf`](./final_results/Project_Report.pdf) | **Main Manuscript:** Complete empirical study across 84 configurations $\times$ 5 seeds (420 runs on Flair2 cluster), 4 core findings, 5-attack suite, and publication figures. |
| **2. Technical Note (8 pp.)** | [`Technical_Note.pdf`](./final_results/Technical_Note.pdf) | **Mathematical Companion:** Full proofs of Theorem 1 (Pre-Clip Weight Erasure), Corollary 1 (Interior Clipping Optimum $C^\star$), Proposition 1 (CTGR-DP-DRO Sensitivity & Majority SNR Trade-Off), and Theorem 2 (Dual-Channel Rényi Accountant). |
| **3. Unified Audit Runner (v5)** | [`Unified_Privacy_Audit_Runner_v5.ipynb`](./final_results/Unified_Privacy_Audit_Runner_v5.ipynb) | **Self-Contained Verification Pipeline:** End-to-end single-run privacy and fairness audit notebook implementing all 7 methods, Dual-Channel RDP accounting, and 5 attacks (Steinke Canary, Shadow MIA, LOO Parametric LiRA, White-Box Gradient Ratio, Counterfactual Memorization). |

---

## 💡 Overview for New Readers: What Is This Project About?

Machine learning models trained via standard **Empirical Risk Minimization (ERM)** often achieve high average accuracy while performing poorly on underrepresented demographic groups. To fix this, practitioners frequently use **Distributionally Robust Optimization (DRO)**—such as $\chi^2$-DRO or CVaR-DRO—which automatically upweights high-loss training examples during optimization.

However, upweighting hard examples raises two fundamental questions:

1. **The Privacy Question:** When a fairness algorithm repeatedly amplifies the gradients of minority or outlier records, does it force the neural network to *memorize* those specific individuals, making them vulnerable to **Membership Inference Attacks (MIAs)**?
2. **The Differential Privacy Question:** If we wrap DRO inside **Differentially Private SGD (DP-SGD)** to protect privacy, does per-sample gradient clipping preserve DRO's fairness gains, or does clipping mathematically cancel out the upweighting?

### What We Proved (After 6 Months of Experimentation)

1. **Instance-Level DRO Creates a Severe Tail-Privacy Asymmetry:** In non-private training ($\varepsilon = \infty$), $\chi^2$-DRO cuts the racial accuracy gap on Adult by **45.3%** ($14.80\% \to 8.10\%$), but inflates **minority tail membership leakage (LiRA TPR @ 0.1% FPR) by $6.2\times$** ($1.11\% \to 6.89\%$). Crucially, this is *not* an inevitable cost of fairness: **Matched-ERM** ($1.89\%$ TPR, $8.10\%$ gap) and **Group-DRO** ($2.01\%$ TPR, $7.40\%$ gap) achieve equal or better fairness with **$3.4\times$–$3.6\times$ lower minority tail leakage**.

2. **Naive Pre-Clip DP-SGD Mathematically Erases DRO Weights:** Under standard pre-clip DP-SGD ($\tilde{g}_i = g_i \cdot \min(1,\, C / \lVert w_i g_i \rVert_2)$), whenever an unclipped minority gradient already exceeds the clipping norm ($\lVert g_i \rVert_2 \ge C$), multiplying by $w_i > 1$ simply scales a vector that gets projected right back onto the radius-$C$ sphere:

   $$\mathrm{Clip}(w_i g_i,\, C) = C \cdot \frac{g_i}{\lVert g_i \rVert_2} = \mathrm{Clip}(g_i,\, C)$$

   At $\varepsilon = 4$ ($C = 1.0$), **88%–99.8%** of minority gradients saturate the clipping threshold across all three datasets, collapsing $\chi^2$-DRO's effective minority weight ratio from **$2.96\times$–$3.17\times$ down to $1.02\times$–$1.03\times$** and reverting the Adult racial gap to **16.10%** (virtually identical to ERM's **16.80%**).

3. **Post-Clip Subgroup Rescaling (CTGR-DP-DRO) Restores Fairness Under Strict DP—With an Explicit Utility Trade-Off:** By clipping individual gradients to radius $C$ *first* and applying bounded subgroup weights $w_{g_i} \in [1, W_{\max}]$ *after* clipping (normalized by $W_{\max}$ so worst-case $\ell_2$-sensitivity remains bounded by $\Delta_2 \le C$, and accounting for epoch-level group-loss queries via a **Dual-Channel Rényi Accountant**), **CTGR-DP-DRO** ($W_{\max}=2.5$) preserves a **$2.74\times$ effective minority weight ratio** at $\varepsilon = 4.00$. It reduces the Adult racial gap from **16.80% to 9.40%** and boosts worst-group accuracy by **+8.2 percentage points** ($65.40\% \to 73.60\%$), at a transparent cost of **$-1.9$ percentage points** in overall accuracy ($76.60\% \to 74.70\%$) due to reduced majority-group gradient SNR ($1.18 \to 0.75$).

---

## 📊 Summary of Empirical Findings (420 Primary Runs across 3 Datasets)

All headline benchmarks in the Project Report reflect **84 configurations** ($3\text{ datasets} \times 7\text{ methods} \times 4\text{ privacy budgets } \varepsilon \in \{\infty, 8, 4, 2\}$ at $\delta = 10^{-5}$) evaluated across **5 random seeds (`2024–2028`)** on the IITGN **Flair2** cluster ($2\times$ NVIDIA L40S GPUs, ~335 GPU-hours).

### Adult Census Benchmark ($N = 32{,}561$, 5 Racial Subgroups)

| Method | Budget $\varepsilon_{\text{total}}$ | Aux Budget $\varepsilon_{\text{aux}}$ | Test Acc (%) | Worst-Group Acc (%) | Race Gap (%) $\downarrow$ | Minority LiRA TPR@0.1%FPR (%) $\downarrow$ | Counterfactual Memorization $\downarrow$ | Effective Weight Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ERM** | $\infty$ | — | $85.20 \pm 0.18$ | $70.40 \pm 0.62$ | $14.80 \pm 0.58$ | $1.11 \pm 0.14$ | $0.18 \pm 0.03$ | $1.00\times$ |
| **$\chi^2$-DRO** | $\infty$ | — | $83.40 \pm 0.24$ | $75.30 \pm 0.55$ | $8.10 \pm 0.49$ | **$6.89 \pm 0.52$** | **$0.84 \pm 0.04$** | $3.12\times$ |
| **CVaR-DRO** | $\infty$ | — | $83.10 \pm 0.27$ | $74.50 \pm 0.61$ | $8.60 \pm 0.54$ | **$6.45 \pm 0.49$** | **$0.81 \pm 0.04$** | $2.98\times$ |
| **Group-DRO** | $\infty$ | — | $83.80 \pm 0.21$ | $76.40 \pm 0.48$ | $7.40 \pm 0.44$ | $2.01 \pm 0.22$ | $0.34 \pm 0.03$ | $2.85\times$ |
| **Adv-Debias** | $\infty$ | — | $83.50 \pm 0.29$ | $74.60 \pm 0.64$ | $8.90 \pm 0.57$ | $2.15 \pm 0.25$ | $0.36 \pm 0.04$ | $1.00\times$ |
| **Matched-ERM** | $\infty$ | — | $83.40 \pm 0.22$ | $75.30 \pm 0.51$ | $8.10 \pm 0.47$ | **$1.89 \pm 0.20$** | $0.31 \pm 0.03$ | $2.76\times$ |
| **CTGR-DP-DRO** | $\infty$ | — | $83.70 \pm 0.21$ | $76.10 \pm 0.49$ | $7.60 \pm 0.45$ | $2.08 \pm 0.23$ | $0.35 \pm 0.03$ | $2.48\times$ |
| **ERM** | $4.00$ | $0.00$ | $76.60 \pm 0.34$ | $65.40 \pm 0.88$ | $16.80 \pm 0.82$ | $0.20 \pm 0.05$ | $0.05 \pm 0.02$ | $1.00\times$ |
| **$\chi^2$-DRO (Pre-Clip)** | $4.00$ | $0.24$ | $76.10 \pm 0.38$ | $65.80 \pm 0.91$ | **$16.10 \pm 0.86$** | $0.21 \pm 0.05$ | $0.06 \pm 0.02$ | **$1.02\times$** |
| **Group-DRO (Pre-Clip)** | $4.00$ | $0.24$ | $76.30 \pm 0.35$ | $66.20 \pm 0.87$ | $15.60 \pm 0.80$ | $0.20 \pm 0.05$ | $0.05 \pm 0.02$ | $1.03\times$ |
| **CTGR-DP-DRO (Ours)** | $4.00$ | $0.24$ | $74.70 \pm 0.41$ | **$73.60 \pm 0.72$** | **$9.40 \pm 0.65$** | $0.21 \pm 0.05$ | $0.06 \pm 0.02$ | **$2.74\times$** |

*(Full tables across all 4 privacy budgets $\varepsilon \in \{\infty, 8, 4, 2\}$ and all 3 datasets—Adult, Texas-100X, and UTKFace—are provided in Table 1 of [`final_results/Project_Report.pdf`](./final_results/Project_Report.pdf).)*

### Subgroup Scaling Law ($K = 20$ Intersectional Subgroups per Dataset, 60 Pooled)

* **Sample Size Scaling:** Across $K=20$ intersectional $(\text{Race} \times \text{Target})$ subgroups per dataset, $\log_{10}(n_g)$ strongly predicts tail membership leakage under $\chi^2$-DRO ($r = -0.846$ on Adult, $-0.862$ on Texas-100X, $-0.838$ on UTKFace; Mixed-Effects Linear Model fixed-effect slope $\beta_1 = -3.41 \pm 0.48$ pp per decade, $p < 10^{-43}$).
* **Subgroup Gradient Concentration Index ($\Psi_g = \bar{w}_g / \sqrt{n_g}$):** Combining mean subgroup optimization weight $\bar{w}_g$ with effective sample size $\sqrt{n_g}$ explains **$R^2 = 93.8\%$–$94.8\%$** of tail-risk variance across all three datasets.

---

## 🕰️ Research Timeline & Repository Evolution (May–June & Sep–Nov)

In the spirit of transparent and reproducible science, **we have not deleted our intermediate failures, negative results, or superseded codebases**. Instead, the folders in this repository are named chronologically so readers can trace how early hypotheses failed and led to the final theoretical and empirical results:

```text
privacy-audit-dro-dpsgd/
│
├── final_results/                                      # ⭐ FINAL VALIDATED DELIVERABLES (START HERE)
│   ├── Project_Report.pdf                              # 15-page Final Report (Flair2 420-run benchmark)
│   ├── Technical_Note.pdf                              # 8-page Mathematical Proofs & Derivations
│   └── Unified_Privacy_Audit_Runner_v5.ipynb           # Self-contained v5 Single-Run Audit Pipeline
│
├── iter1_may_june_initial_explorations_archive/        # Phase 1 (May–June): Early tabular MIA baselines
├── iter1_may_june_modular_src_prototype/               # Phase 1 (May–June): Initial modular PyTorch scripts
│                                                       #   -> Lesson learned: Global loss-threshold MIAs masked
│                                                       #      minority tail risk; motivated per-example LiRA @ 0.1% FPR.
│
├── iter2_sep_preliminary_audit_reports/                # Phase 2 (September): First DRO + DP-SGD stacking runs
│                                                       #   -> Failure uncovered: Naive DP-SGD + DRO destroyed fairness
│                                                       #      at ε ≤ 4, sparking the Pre-Clip Weight Erasure analysis.
│
├── iter3_oct_dataset_sweep_notebooks_v1_v4/            # Phase 3 (October): Multi-dataset sweeps (v1–v4 notebooks)
├── iter3_oct_figure_extraction_scripts/                # Phase 3 (October): Early plotting & extraction utilities
│                                                       #   -> Failure uncovered: Unnormalized post-clip reweighting
│                                                       #      inflated sensitivity (Δ₂ = W_max * C) and ignored
│                                                       #      epoch-level group-loss privacy leakage (Channel 2).
│
├── iter4_nov_early_manuscript_drafts/                  # Phase 4 (November): Pre-cluster drafts & figure prototypes
├── iter4_nov_editorial_illustrations_docs/             # Phase 4 (November): Conceptual diagrams for presentations
├── iter4_nov_worklogs_and_draft_notes/                 # Phase 4 (November): Development logs & intermediate notes
│                                                       #   -> Resolution: Formulated Dual-Channel Rényi Accountant,
│                                                       #      normalized CTGR-DP-DRO (Δ₂ ≤ C), LOO Parametric LiRA,
│                                                       #      and executed the 5-seed Flair2 cluster campaign.
│
├── README.md                                           # Project overview & documentation (this file)
└── requirements.txt                                    # Python dependencies for the v5 Audit Runner

### Key Lessons From Past Iterations
1. **Why Early Average-Case MIAs Failed (`iter1_may_june_*`):** Initially, we evaluated privacy using global balanced accuracy and standard loss-threshold attacks. Because minority subgroups make up $<5\%$ of the dataset, global MIA metrics showed almost no difference between ERM and DRO. Moving to **subgroup-stratified LiRA TPR @ 0.1% FPR** and **Counterfactual Label-Noise Canaries** exposed the $6.2\times$ tail vulnerability.
2. **Why Naive DP-SGD + DRO Failed (`iter2_sep_*`):** When we first combined Opacus DP-SGD with $\chi^2$-DRO, fairness gains completely vanished at $\varepsilon = 4$ and $\varepsilon = 2$. Diagnosing per-sample gradient norms revealed that **88%–99.8% of minority gradients were already above $C=1.0$ before upweighting**, leading to **Theorem 1 (Pre-Clip Weight Erasure)**.
3. **Why First-Generation Post-Clip Reweighting Was Flawed (`iter3_oct_*`):** Our initial attempt to rescale gradients after clipping multiplied clipped vectors by $w_g \in [1, W_{\max}]$ *without* dividing by $W_{\max}$ and *without* accounting for the epoch-level group-loss queries used to update $w_g$. Under rigorous DP auditing, unnormalized rescaling inflates sensitivity to $\Delta_2 = W_{\max} C$ (pushing true $\varepsilon$ from $4.0$ to $>9.4$). In our final **v5 pipeline and Report**, we normalize updates by $W_{\max}$ ($\Delta_2 \le C$) and compose both channels explicitly via the **Dual-Channel Rényi Accountant** ($\varepsilon_{\text{total}} = \varepsilon_{\text{grad}} \oplus_{\text{RDP}} \varepsilon_{\text{aux}}$).

---

## 🚀 Quick Start: Running the Final v5 Audit Notebook

To run the self-contained verification pipeline ([`final_results/Unified_Privacy_Audit_Runner_v5.ipynb`](./final_results/Unified_Privacy_Audit_Runner_v5.ipynb)):

```bash
# 1. Clone the repository
git clone https://github.com/Govindu1729/privacy-audit-dro-dpsgd.git
cd privacy-audit-dro-dpsgd

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter and open the final v5 notebook
jupyter notebook final_results/Unified_Privacy_Audit_Runner_v5.ipynb
```

### Choosing the Dataset in Cell 2 of the v5 Notebook
In Cell 2, set `ACTIVE_DATASET` to audit any of the three benchmarks:
* `ACTIVE_DATASET = "adult"` — UCI Adult Census ($N = 32,561$, 5 racial subgroups, classification)
* `ACTIVE_DATASET = "texas"` — Texas-100X Hospital Discharge ($N = 50,000$ stratified single-run audit subsample of the $N = 99,492$ cohort, 5 racial/ethnic subgroups, classification)
* `ACTIVE_DATASET = "utkface"` — UTKFace Facial Age Benchmark ($N = 23,705$, 4 racial subgroups $\times$ 5 age bands, regression)

---

## 📚 Key References

1. **Abadi et al. (2016).** *Deep Learning with Differential Privacy.* ACM CCS 2016.
2. **Hashimoto et al. (2018).** *Fairness Without Demographics in Repeated Loss Minimization.* ICML 2018.
3. **Sagawa et al. (2020).** *Distributionally Robust Neural Networks for Group Shifts.* ICLR 2020.
4. **Carlini et al. (2022).** *Membership Inference Attacks From First Principles (LiRA).* IEEE S&P 2022.
5. **Steinke, Nasr & Jagielski (2023).** *Privacy Auditing with One (1) Training Run.* NeurIPS 2023.
6. **Bagdasaryan, Poursaeed & Shmatikov (2019).** *Differential Privacy Has Disparate Impact on Model Accuracy.* NeurIPS 2019.
