# Sparsity Is What You Need: Multi-scale Sparse Attention for Multi-regime Time Series: Code and sample data


## Layout

```
sparsity-code/
  code/                 model, training pipeline, feature engineering, experiment drivers
    model.py            MISA (class name MSCABiGRU): 3-scale BiGRU + sparsemax temporal/scale attention + gated cross-asset MHA
    sparsemax.py        sparsemax projection
    features.py         OHLCV -> 7 features (two-stage correlation filter + PCA ranking)
    train.py            Trainer, NMSE, Harvey-corrected Diebold-Mariano test
    run_experiment.py   one asset-period pair (target + leader CSV)
    run_multiseed.py    seeds 42/43/44 on the 15 representative pairs
    run_52assets.py     all 47 assets x 3 regimes (139 pairs), LSTM/GRU baselines
    run_ablation.py     ABL-A softmax, ABL-B single scale, ABL-C no gate
    run_transformer_dm*.py, run_informer_dm.py   iTransformer / PatchTST / Informer, re-trained on the same data
    run_garch.py, run_dcc_garch*.py              AR(1)-GARCH(1,1) reference, DCC-GARCH correlations
    run_lnn.py          LNN/CfC baseline
    run_complexity.py, run_gradient_efficiency.py, run_extended_baselines.py
    vol139/             realized-variance study (Section 6.6)
      vol139.py         trains MISA/LSTM/GRU on log RV with the QLIKE loss; fits naive, EWMA, HAR-RV, HAR-X, GARCH references
      harx_ql.py        HAR-RV / HAR-X fitted by QLIKE
      agg139.py         aggregates per-pair results into the paper's tables (DM on QLIKE differentials)
      agg_abl.py        aggregates the ablation runs
    pyproject.toml, README.md
  data/                 sample 5-minute Coinbase OHLCV bars (18 of the 141 files used)
    {BTC,ETH,XRP,LTC,DOGE,BCH}_5m_{P1_20221101,P2_20231001,P3_20240301}.csv
    download_expanded.py, download_sol_ada.py, download_doge_real.py   collection scripts (public Coinbase Exchange REST API, no key)
  results/              per-pair numbers behind the tables
    expanded_results.json       139 pairs: NMSE and DM vs LSTM/GRU (file stores DM with NEGATIVE = MISA better; the paper prints the opposite sign)
    multiseed_results.json      15 representative pairs, seeds 42/43/44
    transformer_dm_results.json iTransformer / PatchTST / Informer DM
    garch_results.json, lnn_results.json, dcc_garch_52assets.json, ablation_results.json
    random_walk_baseline.json   no-change forecast NMSE for the 139 pairs
    fx_dm_results.json          foreign-exchange study
    vol139/summary.json, summary_ablation.json, table_vol_main.json   realized-variance study
```

## Running

```bash
cd code
pip install -e .                      # installs the msca_sparsity package (PyTorch >= 2.0, numpy, pandas, scikit-learn, scipy, arch)
python run_experiment.py --tgt ../data/ETH_5m_P2_20231001.csv --ca ../data/BTC_5m_P2_20231001.csv
python run_multiseed.py               # Table 2 (15 pairs, 3 seeds)
python run_52assets.py                # Table S5 (139 pairs)
python vol139/vol139.py ETH --seed 42 # realized-variance study for one asset (edit DATA/--out paths at the top of the script)
```

Period windows: P1 = November 2022 (FTX crisis), P2 = October 2023 (bull market), P3 = March 2024 (post-ETF).
Each 30-day window is split 70/10/20 into train/validation/test; features and scalers are fitted on the training part only.

The remaining 123 data files are produced by `data/download_expanded.py` (public endpoint `https://api.exchange.coinbase.com`, granularity 300 s).
