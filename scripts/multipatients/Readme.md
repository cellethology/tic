# Multi-patients Bootstrap Trials – README  
*Location → `scripts/multipatients/`*

This document explains **what each script does, the expected inputs / outputs, and how to reproduce the visualisations** shown in  
`notebooks/visualization/plot_boostrap_trials.ipynb`.

---

## 1  Pipeline at a Glance

```
┌───────────────────────────────────────────────┐
│ 1. extract_feature_adata.py                  │  ➜  feature_adata_<dataset>.h5ad
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│ 2. bootstrap.py                              │  ➜  bootstrap_trials/trial_XXX.h5ad
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│ 3. causal_infer_posthoc.py                   │  ➜  bootstrap_trials/<Outcome>/trial_XXX_causal.csv
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│ 4. plot_boostrap_trials.ipynb (notebook)     │  ➜  Figures / tables
└───────────────────────────────────────────────┘
```

---

## 2  Step-by-Step Usage

### 2-1  Feature extraction per region

```bash
python scripts/multipatients/extract_feature_adata.py \
       --dataset dfci \
       --out_dir results/features \
       --max_regions 40               # optional
```

*Inputs*

* `tic.data.codex.loader:list_regions` must find raw region folders.

*Outputs*

* `results/features/feature_adata_dfci.h5ad`
* `results/features/feature_adata_dfci_m_fraction.csv`
* a PNG histogram of mesenchymal fractions.

---

### 2-2  Bootstrap trials

```bash
python scripts/multipatients/bootstrap.py \
       results/features/feature_adata_dfci.h5ad \
       --rep_key composition \
       --n_trials 100 \
       --sample_size 80000 \
       --exp_root results/multipatients
```

*Creates* `results/multipatients/bootstrap_trials/trial_000.h5ad … trial_099.h5ad`  
plus a `manifest.json`.

---

### 2-3  Post-hoc causal inference

Repeat once for **each outcome gene** you care about (e.g. PanCK, Vimentin):

```bash
python scripts/multipatients/causal_infer_posthoc.py \
       --trials_dir results/multipatients/bootstrap_trials \
       --outcome PanCK \
       --prior_trend decrease \
       --bins 100
```

This adds a new sub-folder, e.g.

```
results/multipatients/bootstrap_trials/PanCK/
└── trial_000_causal.csv
```

---

### 2-4  Visualisation notebook

Open:

```
notebooks/visualization/plot_boostrap_trials.ipynb
```

1. Point the **PanCK** and **Vimentin** directory paths (first code cell).  
2. Run *all cells*: you will get  

   * A table listing predictors significant in **both** outcomes  
   * A bar-plot with ±1 σ error bars and significance stars, sorted by EMT-direction score (Vim – PanCK).

---

## 3  Typical Folder Layout

```
project/
├─ raw_data/                       # your original region folders
├─ scripts/
│   └─ multipatients/              # (this directory)
├─ notebooks/
│   └─ visualization/
│       └─ plot_boostrap_trials.ipynb
└─ results/
    ├─ features/
    │   └─ feature_adata_dfci.h5ad
    └─ multipatients/
        └─ bootstrap_trials/
            ├─ trial_000.h5ad
            ├─ …
            ├─ PanCK/
            │   └─ trial_000_causal.csv
            └─ Vimentin/
                └─ trial_000_causal.csv
```

