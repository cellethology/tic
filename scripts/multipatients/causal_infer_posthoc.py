#!/usr/bin/env python
"""Run causal inference *after* bootstrapping, using existing .h5ad trials.

Usage:
------
python scripts/multipatients/causal_infer_posthoc.py \
    --trials_dir ../results/multipatients/bootstrap_trials \
    --outcome PanCK \
    --bins 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc
from tqdm import tqdm
from tic.utils import set_random_seed
from tic.wrappers.causal import CausalWrapper
from tic.utils.logging import get_logger

LOGGER = get_logger(__name__)


def run_posthoc_causal(
    trials_dir: Path,
    outcome: str,
    prior_trend: str,
    bins: int = 100,
    feature_key: str = "X_predictors",
    include_extractors: list[str] = ["composition"],
    seed: int = 42,
):  
    set_random_seed(seed=seed)
    
    trial_files = sorted(trials_dir.glob("trial_*.h5ad"))
    if not trial_files:
        raise FileNotFoundError(f"No trial_*.h5ad files found in {trials_dir}")

    out_dir = trials_dir / outcome
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in tqdm(trial_files, desc="Post-hoc causal inference"):
        adata = sc.read_h5ad(f)

        cw = CausalWrapper(
            outcome=outcome,
            feature_key=feature_key,
            include_extractors=include_extractors,
            method="granger_causality",
            prior_trend=prior_trend,
            bins=bins,
        )
        cw.fit(adata)
        df = pd.DataFrame.from_dict(cw.results, orient="index")
        out_csv = out_dir / f"{f.stem}_causal.csv"
        df.to_csv(out_csv)

        LOGGER.info("Saved: %s", out_csv)


def build_parser():
    parser = argparse.ArgumentParser(description="Run causal inference on bootstrap trials.")
    parser.add_argument("--trials_dir", type=Path, required=True,
                        help="Directory containing trial_XXX.h5ad files")
    parser.add_argument("--outcome", type=str, required=True,
                        help="Outcome gene for causal inference")
    parser.add_argument("--prior_trend", type=str, choices=['none', 'increase', 'decrease'], required=True,
                        help="prior trend for the outcome varible in the transition process")
    parser.add_argument("--bins", type=int, default=100,
                        help="# bins for causal analysis")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    return parser


def main():
    args = build_parser().parse_args()

    run_posthoc_causal(
        trials_dir=args.trials_dir,
        outcome=args.outcome,
        prior_trend=args.prior_trend,
        bins=args.bins,
        seed=args.seed,
    )

    LOGGER.info("All done.")


if __name__ == "__main__":
    main()