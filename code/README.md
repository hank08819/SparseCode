> **Note.** The package and class names below (`msca_sparsity`, `MSCABiGRU`) are the original code names. In the Pattern Recognition manuscript the model is called **MISA** (Multi-scale Inter-signal Sparse Attention). The architecture is identical.

# MSCA-Sparsity

**Multi-Scale Cross-Asset Sparse Attention for Cryptocurrency Price Discovery**

Python package for the MSCA-SBiGRU model published in:

> H. Han, "Sparsity Is All You Need in Multi-Scale Cross-Asset Attention for
> Cryptocurrency Price Discovery," *Expert Systems with Applications*, 2026.

---

## Model Summary

MSCA-SBiGRU combines three architectural ideas into a single sparse recurrent model:

| Component | Mechanism | Effect |
|-----------|-----------|--------|
| Multi-scale encoding | BiGRU at k=1 (5-min), k=3 (15-min), k=6 (30-min) | Captures intra-hour structure |
| Sparsemax temporal attention | L₂-projection onto simplex; exact zeros | Focuses on momentum bars, ignores noise |
| Cross-asset attention | MHA with binary gate regularisation λ·g(1−g) | Exploits BTC price leadership |

**Key result:** 6 statistically significant wins vs LSTM/GRU (DM test, Harvey 1997 correction) across 9 asset–period pairs spanning 3 market regimes. First deep-learning model to achieve DM-validated superiority on cryptocurrency data.

---

## Installation

```bash
cd msca_sparsity
pip install -e .
```

**Dependencies:** Python ≥ 3.9, PyTorch ≥ 2.0, NumPy, Pandas, scikit-learn, SciPy, Matplotlib

---

## Quick Start

```python
from msca_sparsity import MSCABiGRU, Trainer, build_features

# Load and engineer features
close, feat_tgt = build_features("data/BTC_5m_P1.csv", n_features=7)
_,     feat_ca  = build_features("data/ETH_5m_P1.csv", n_features=7)

# Build model and train
model   = MSCABiGRU(feat_dim=7, hidden=64, n_heads=4, dropout=0.25)
trainer = Trainer(model, lr=5e-4, epochs=200, patience=25)
trainer.fit(X_train, Xca_train, y_train, X_val, Xca_val, y_val)

# Evaluate
metrics = trainer.evaluate(X_test, Xca_test, y_test,
                            close_prices=pte, price_range=price_range)
print(metrics)  # {"NMSE": ..., "RMSE": ..., "MAPE": ..., "R2": ...}
```

---

## Package Structure

```
msca_sparsity/
├── pyproject.toml
├── README.md
├── examples/
│   ├── run_experiment.py     # end-to-end training + ablation
│   └── generate_figures.py  # reproduce all paper figures
├── figures/                  # output PDFs land here
└── msca_sparsity/
    ├── __init__.py
    ├── model.py              # MSCABiGRU architecture
    ├── sparsemax.py          # Sparsemax activation (Martins & Astudillo, 2016)
    ├── features.py           # Feature pipeline (COH, RSI, PCA selection)
    ├── train.py              # Trainer class + DM test
    └── figures/
        ├── __init__.py
        └── plots.py          # All paper figures (matplotlib)
```

---

## Ablation Variants

```python
# Full model (A)
MSCABiGRU(use_multiscale=True,  use_cross_asset=True)

# No cross-asset attention (C) — ablation
MSCABiGRU(use_multiscale=True,  use_cross_asset=False)

# No multi-scale encoding (B) — ablation
MSCABiGRU(use_multiscale=False, use_cross_asset=True)
```

---

## Running Examples

```bash
# Full experiment on one asset-period pair
python examples/run_experiment.py \
    --tgt data/BTC_5m_P1_20221101.csv \
    --ca  data/ETH_5m_P1_20221101.csv

# Regenerate all figures from saved results
python examples/generate_figures.py \
    --results ../results/msca_results.json \
    --out figures/
```

---

## Citation

```bibtex
@article{han2026msca,
  author  = {Henry Han},
  title   = {Sparsity Is All You Need in Multi-Scale Cross-Asset Attention
             for Cryptocurrency Price Discovery},
  journal = {Expert Systems with Applications},
  year    = {2026},
}
```

---

## License

MIT — Henry Han, Baylor University (<henry_han@baylor.edu>)
