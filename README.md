# Privacy Auditing Pipeline: DRO vs. DP-SGD

This repository contains the core pipeline for auditing the fairness-privacy trade-off when stacking **Distributionally Robust Optimization (DRO)** with **Differentially Private SGD (DP-SGD)**.

## 🔍 Core Research Question
Does DRO's fairness benefit come at a hidden privacy cost (Membership Inference Attack vulnerability), and does DP-SGD fully mask it? Specifically, we investigate if DP-SGD's gradient clipping erases the magnitude-channel signal that DRO amplifies in minority subgroups.

## 🏗️ Pipeline Architecture
- **Target Models**: Deep Tabular ResNet (Texas/Adult), ResNet-18 with GroupNorm (UTKFace).
- **Training**: DP-SGD with per-ε $\alpha$ calibration and DRO dual-variable updates.
- **Auditing**: Black-Box (LiRA) and White-Box (Gradient Direction) Membership Inference Attacks.
- **Auto-Tuner**: Automated hyperparameter search to ensure models pass the C1/C2/C3 quality rubric (memorization capacity, fairness gap, DRO saturation).

## 📂 Repository Structure
- `notebooks/` - End-to-end sweep runners for Adult, Texas, and UTKFace.
- `src/` - Core data loaders, attack implementations, and DRO/DP-SGD training loops.
- `docs/` - Project dashboard, decision logs, and methodology documentation.

> **Note:** Raw datasets and heavy experiment checkpoints (`.pt` files) are excluded from this repository to maintain a lightweight footprint.
