# tic/plotting/cross_region_heatmap.py
"""
Plot a cross-region causal-effect heatmap with flexible *Top-N predictor* selection.

Implemented strategies
----------------------
* ``"max"``      - absolute maximum effect (legacy behaviour)
* ``"median"``   - median absolute effect across regions
* ``"frequency"``- proportion of regions where predictor is significant
* ``"composite"``- α·median + (1 − α)·frequency  (α configurable)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------------------------- #
#                                    Typing                                      #
# ----------------------------------------------------------------------------- #
SignArg = Union[Literal["positive", "pos", "+"], Literal["negative", "neg", "-"], int]
MetricArg = Literal["max", "mean", "rms", "p_value"]
TopStrategy = Literal["max", "median", "frequency", "composite"]


def _to_sign(val: SignArg | None) -> Optional[int]:
    """Convert a user-friendly sign string to ±1 or None."""
    if val is None:
        return None
    if isinstance(val, int):
        return 1 if val >= 0 else -1
    val = val.lower()
    if val in {"positive", "pos", "+"}:
        return 1
    if val in {"negative", "neg", "-"}:
        return -1
    raise ValueError(f"Unsupported expected_sign: {val}")


# ----------------------------------------------------------------------------- #
#                              Predictor ranking                                 #
# ----------------------------------------------------------------------------- #
def _rank_predictors(
    df: pd.DataFrame,
    strategy: TopStrategy,
    top_n: int,
    alpha: float = 0.5,
) -> list[str]:
    """
    Return a list of the top-N predictor names according to *strategy*.

    Parameters
    ----------
    df
        DataFrame of shape (regions × predictors) with filled zeros for
        “not-significant” cells.
    strategy
        Ranking method.
    top_n
        Number of predictors to keep.
    alpha
        Weight for composite score; ignored for other strategies.

    Returns
    -------
    list[str]
        Ordered predictor names.
    """
    if strategy == "max":
        score = df.abs().max(axis=0)

    elif strategy == "median":
        score = df.abs().median(axis=0)

    elif strategy == "frequency":
        score = (df != 0).sum(axis=0) / df.shape[0]

    elif strategy == "composite":
        median = df.abs().median(axis=0)
        freq = (df != 0).sum(axis=0) / df.shape[0]
        score = alpha * median + (1.0 - alpha) * freq

    else:  # pragma: no cover
        raise ValueError(f"Unknown top_n strategy: {strategy}")

    return score.sort_values(ascending=False).index[:top_n].tolist()


# ----------------------------------------------------------------------------- #
#                               Main plot function                              #
# ----------------------------------------------------------------------------- #
def plot_cross_region_causal_heatmap(
    out_dir: str | Path,
    region_ids: List[str],
    y_var: str,
    metric: MetricArg = "max",
    p_threshold: float = 0.05,
    log2_transform: bool = True,
    expected_sign: SignArg | None = None,
    top_n: int | None = 80,
    top_strategy: TopStrategy = "max",
    composite_alpha: float = 0.5,
    row_cluster: bool = True,
    col_cluster: bool | None = None,
    figsize: tuple[int, int] | None = None,
    cmap: str = "coolwarm",
    title: str | None = None,
    save_path: Optional[str] = None,
) -> Optional[plt.Axes]:
    """
    Plot a (clustered) heatmap of causal coefficients across regions.

    Parameters
    ----------
    out_dir
        Root directory containing region sub-folders with `results.pkl`.
    region_ids
        Regions to include (folder names).
    y_var
        Target biomarker (Y) name.
    metric
        Effect metric: ``"max"``, ``"mean"``, ``"rms"``, or ``"p_value"``.
    p_threshold
        Per-region adjusted-p cutoff for keeping a predictor.
    log2_transform
        Apply signed log2 to effect metrics (ignored for p_value metric).
    expected_sign
        Expected *monotonic* sign of Y’s correlation; rows with opposite sign
        are flipped (EMT ↔ MET correction).
    top_n
        Number of predictors (columns) to keep. ``None`` keeps all.
    top_strategy
        How to rank predictors for Top-N selection
        (“max” | “median” | “frequency” | “composite”).
    composite_alpha
        Weight for *median* in the composite score (0-1).
    row_cluster, col_cluster
        Clustering options passed to seaborn.
    figsize, cmap, title, save_path
        Plot customisation.
    """
    metric = metric.lower()
    if metric not in {"max", "mean", "rms", "p_value"}:
        raise ValueError(f"Unsupported metric: {metric}")

    exp_sign = _to_sign(expected_sign)

    # ----------------------------------------------------------------- collect
    rows: Dict[str, Dict[str, float]] = {}
    sig_map: Dict[tuple[str, str], bool] = {}

    for rid in region_ids:
        rdir = Path(out_dir) / rid
        pkl_path = rdir / "causal" / y_var / "results.pkl"
        if not pkl_path.is_file():
            continue

        # -- direction flip
        flip = False
        if exp_sign is not None:
            metrics_csv = rdir / "metrics.csv"
            if metrics_csv.is_file():
                corr = (
                    pd.read_csv(metrics_csv)
                    .query("biomarker == @y_var & mono_method == 'spearman'")
                    .get("mono_correlation")
                )
                if not corr.empty:
                    sign = (
                        np.sign(corr.iloc[0])
                        if not np.isclose(corr.iloc[0], 0)
                        else 0
                    )
                    flip = sign != 0 and sign != exp_sign

        with pkl_path.open("rb") as fh:
            results: Dict[str, dict] = pickle.load(fh)

        row: Dict[str, float] = {}
        for pred, stats in results.items():
            if stats["p_value"] > p_threshold:
                continue

            if metric == "p_value":
                val = -np.log10(stats["p_value"] + 1e-12)
            else:
                coeffs = np.asarray(stats["best_model_parameters"], float)
                if metric == "max":
                    val = coeffs[np.argmax(np.abs(coeffs))]
                elif metric == "mean":
                    val = float(np.nanmean(coeffs))
                else:  # rms
                    val = np.sign(np.nanmean(coeffs)) * np.sqrt(
                        np.nanmean(coeffs**2)
                    )
                if log2_transform:
                    val = np.sign(val) * np.log2(np.abs(val) + 1e-12)

            row[pred] = -val if flip else val
            sig_map[(rid, pred)] = True

        if row:
            rows[rid] = row

    if not rows:
        print("[WARN] No significant predictors found.")
        return None

    # ----------------------------------------------------------------- DataFrame
    df = (
        pd.DataFrame.from_dict(rows, orient="index")
        .reindex(sorted({c for r in rows.values() for c in r}), axis=1)
        .fillna(0.0)
    )

    if top_n and df.shape[1] > top_n:
        keep = _rank_predictors(df, top_strategy, top_n, composite_alpha)
        df = df[keep]

    n_rows, n_cols = df.shape
    vmax = np.nanpercentile(np.abs(df.values), 99)

    # ----------------------------------------------------------------- Plotting
    if col_cluster is None:
        col_cluster = n_cols <= 150

    if figsize is None:
        figsize = (max(12, 0.08 * n_cols), max(8, 0.25 * n_rows))

    sns.set_theme(style="white")

    cg = sns.clustermap(
        df,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.3,
        figsize=figsize,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": _cbar_label(metric, log2_transform)},
    )
    ax = cg.ax_heatmap

    _set_axis_labels(ax, df, n_rows, n_cols)

    if not title:
        title = (
            f"Causal effect on '{y_var}' "
            f"(metric={metric}, p≤{p_threshold}, top_{top_strategy}={top_n})"
        )
    ax.set_title(title, pad=40)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        return None

    return ax


# ----------------------------------------------------------------------------- #
#                               Helper functions                                #
# ----------------------------------------------------------------------------- #
def _cbar_label(metric: str, log2_transform: bool) -> str:
    """Return colour-bar label."""
    if metric == "p_value":
        return "-log10(adj p)"
    return f"{metric} ({'log2' if log2_transform else 'raw'})"


def _set_axis_labels(ax: plt.Axes, df: pd.DataFrame, n_rows: int, n_cols: int) -> None:
    """Smart axis labelling to avoid clutter."""
    # X (predictors)
    if n_cols <= 80:
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_xticklabels(
            df.columns, rotation=90, fontsize=max(4, 9 - int(np.log10(n_cols)))
        )
    elif n_cols <= 200:
        step = int(np.ceil(n_cols / 80))
        ticks = np.arange(0, n_cols, step) + 0.5
        ax.set_xticks(ticks)
        ax.set_xticklabels(df.columns[::step], rotation=90, fontsize=4)
    else:
        ax.set_xticks([])
        ax.set_xlabel(f"Predictors (n={n_cols})", fontsize=10)

    # Y (regions)
    if n_rows <= 60:
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_yticklabels(
            df.index, fontsize=max(4, 9 - int(np.log10(n_rows)))
        )
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"Regions (n={n_rows})", fontsize=10)