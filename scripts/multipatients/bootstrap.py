#!/usr/bin/env python
"""scripts/multipatients/bootstrap.py

Bootstrap analysis of tumour micro-environment representations.

*Input*
-------
A **feature AnnData** file produced by
``scripts/multipatients/extract_feature_adata.py``.  Each observation (row) is
one tumour-centred micro-environment with all predictor matrices already
constructed.

*Workflow per trial*
--------------------
1. **Sampling** – draw *k* micro-environments (with replacement if population <
   *k*).
2. **Pseudo-time** – compute TIC ordering via :class:`PseudotimeWrapper`.
3. **Causal inference** (optional) – run :class:`CausalWrapper` on a user-chosen
   outcome gene.

All output lives under
``<exp_root>/<dataset>/bootstrap_trials/trial_XXX.{h5ad,csv}``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import scanpy as sc
from tqdm import tqdm

from tic.utils import set_random_seed
from tic.wrappers import PseudotimeWrapper
from tic.utils.logging import get_logger

###############################################################################
# ------------------------- Constants & helpers ----------------------------- #
###############################################################################
LOGGER = get_logger(__name__)


UNS_KEYS_TO_KEEP: Sequence[str] = (
    "feature_modes",
    "graph_params",
    "subgraph_params",
    "X_predictors_names",
    "pseudotime_metrics",
    "pipeline_config",
)

def run_bootstrap(
    adata: sc.AnnData,
    rep_key: str,
    n_trials: int,
    sample_size: int,
    out_dir: Path,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    print(adata)
    n_obs = adata.n_obs
    replace = n_obs < sample_size
    sample_size = min(sample_size, n_obs)

    out_dir.mkdir(parents=True, exist_ok=True)

    log_meta: list[dict] = []  # for a summary json

    for i in tqdm(range(n_trials), desc="Bootstrap trials"):
        idx = rng.choice(n_obs, size=sample_size, replace=replace)
        sub = adata[idx].copy()

        # TIC pseudo-time
        pt_wrap = PseudotimeWrapper(rep_key=rep_key)
        # Remove NaNs in .obsm[rep_key]
        rep_matrix = sub.obsm[rep_key]
        valid_mask = ~np.isnan(rep_matrix).any(axis=1)
        sub = sub[valid_mask].copy()

        if sub.n_obs < 2:
            LOGGER.warning("Trial %03d skipped due to insufficient non-NaN samples", i)
            continue

        pt_adata = pt_wrap.fit(sub)
        pt_adata.write_h5ad(out_dir / f"trial_{i:03d}.h5ad")

        entry = {"trial": i, "pt_h5ad": str(out_dir / f"trial_{i:03d}.h5ad")}       

        log_meta.append(entry)

    # write manifest
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(log_meta, indent=2))

###############################################################################
# ------------------------------ CLI ---------------------------------------- #
###############################################################################

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bootstrap TIC & causal analysis")
    p.add_argument("feature_h5ad", type=Path, help="Path to feature AnnData (.h5ad)")
    p.add_argument("--rep_key", type=str, default="composition", choices=["composition", "centre_gene","centre_gene_comp"], help="Representation key for pseudotime inference ,stored in adata.obsm")
    p.add_argument("--n_trials", type=int, default=100, help="# bootstrap trials")
    p.add_argument("--sample_size", type=int, default=100_000,
                   help="Micro-environments per trial (default: %(default)s)")          
    p.add_argument("--exp_root", type=Path, default=Path("../results/multipatients"),
                   help="Directory root for outputs")
    p.add_argument("--seed", type=int, default=42, help="Global random seed")
    return p


def main() -> None:
    args = build_parser().parse_args()

    set_random_seed(args.seed)

    # Load feature AnnData
    feat_adata = sc.read_h5ad(args.feature_h5ad)
    out_dir = args.exp_root / "bootstrap_trials"

    run_bootstrap(
        adata=feat_adata,
        rep_key=args.rep_key,
        n_trials=args.n_trials,
        sample_size=args.sample_size,
        out_dir=out_dir,
        seed=args.seed,
    )
    LOGGER.info("All done - results written to %s", out_dir)

if __name__ == "__main__":
    main()
