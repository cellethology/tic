# tic/plotting/multipatients.py
"""Visualization utilities for multi-patient bootstrap & causal-inference results.

Key entry points
----------------
plot_causal_single  – bar-plot of −log₁₀(p) for one trial (CSV).
plot_causal_trials  – binary heat-map of significance across many trials.

Both functions return the *matplotlib* Axes object to enable further tweaking
or saving by the caller.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal, Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# seaborn provides prettier heat-maps; fall back gracefully if unavailable
try:
    import seaborn as sns  # type: ignore
except ImportError:  # pragma: no cover
    sns = None  # pytype: disable=annotation-type-mismatch

__all__ = [
    "load_causal_csv",
    "plot_causal_single",
    "plot_causal_trials",
    "plot_causal_trials_custom_effect"
]
EffectMetric = Literal["direct", "mean", "rms", "max"]
SignArg = Literal[1, -1]

###############################################################################
# ----------------------------- I/O helpers --------------------------------- #
###############################################################################

def load_causal_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a *trial_XXX_causal.csv* file and return a tidy DataFrame.

    Adds two convenience columns:
    * ``predictor``   – original index preserved as a column.
    * ``clean_name``  – index stripped of the ``composition:`` prefix for
      cleaner plotting.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, index_col=0)
    df["predictor"] = df.index
    df["clean_name"] = df["predictor"].str.replace(r"^composition:", "", regex=True)
    return df

###############################################################################
# ----------------------------- Main plots ---------------------------------- #
###############################################################################

def plot_causal_single(
    csv_path: str | Path,
    *,
    alpha: float = 0.05,
    top_n: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
):
    """Horizontal bar-plot of −log₁₀(p-value) for a *single* trial.

    Parameters
    ----------
    csv_path
        Path to the ``trial_XXX_causal.csv`` file.
    alpha
        Significance threshold.  A vertical dashed line is drawn at
        *−log₁₀(alpha)*.
    top_n
        If given, plot only the *N* most significant predictors.
    ax
        Target *matplotlib* axes.  If ``None`` a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes  – the axes containing the plot.
    """
    df = load_causal_csv(csv_path).sort_values("p_value")
    if top_n is not None:
        df = df.head(top_n)

    values = -np.log10(df["p_value"].clip(lower=1e-300))  # avoid -inf
    sig = df["p_value"] < alpha

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, max(4, 0.3 * len(df))))

    ax.barh(df["clean_name"], values, color=np.where(sig, "tab:red", "lightgray"))
    ax.set_xlabel("−log₁₀(p-value)")
    ax.set_ylabel("Predictor")
    ax.set_title(Path(csv_path).stem)
    ax.axvline(-np.log10(alpha), linestyle="--", color="k", linewidth=1)
    ax.invert_yaxis()  # most significant on top
    plt.tight_layout()
    return ax


def plot_causal_trials(
    csv_paths: Sequence[str | Path],
    *,
    alpha: float = 0.05,
    ax: Optional[plt.Axes] = None,
):
    """Heat-map of *significant* causal links across many trials.

    A cell is **red** if the predictor is significant (p < alpha) in that trial,
    otherwise white.
    """
    # Aggregate p-values
    dfs: list[pd.Series] = []
    for p in csv_paths:
        df = load_causal_csv(p)
        dfs.append(df["p_value"].rename(Path(p).stem))
    mat = pd.concat(dfs, axis=1)
    mat = mat.sort_index()  # stable ordering across trials
    sig = (mat < alpha).astype(int)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(0.25 * sig.shape[1] + 4, 0.35 * sig.shape[0] + 2)
        )

    if sns is not None:
        sns.heatmap(
            sig,
            cmap="Reds",
            cbar=False,
            ax=ax,
            linewidths=0.5,
            linecolor="silver",
        )
    else:  # fallback to imshow
        ax.imshow(sig, cmap="Reds", aspect="auto")
        ax.set_xticks(range(sig.shape[1]))
        ax.set_xticklabels(sig.columns, rotation=90)
        ax.set_yticks(range(sig.shape[0]))
        ax.set_yticklabels(sig.index)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Predictor")
    ax.set_title(f"Significant causal effects (α = {alpha})")
    plt.tight_layout()
    return ax




def plot_causal_trials_custom_effect(
    csv_paths: Sequence[str | Path],
    *,
    effect_metric: EffectMetric = "direct",
    alpha: float = 0.05,
    expected_sign: SignArg | None = None,
    row_cluster: bool = True,
    figsize: Optional[tuple[int, int]] = None,
    cmap: str = "coolwarm",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Optional[plt.Axes]:
    """
    Plot heatmap of directional causal effect from multiple trials.

    Parameters
    ----------
    csv_paths : list of paths to *_causal.csv
    effect_metric : effect extraction method ["direct", "mean", "rms", "max"]
    alpha : significance threshold
    expected_sign : if given (+1 or -1), flip effect to align direction
    row_cluster : whether to cluster predictors (rows)
    figsize : (w, h) of heatmap
    cmap : color map
    title : plot title
    save_path : if given, save to file instead of showing
    """
    def extract_effect(stats_row: pd.Series) -> Optional[float]:
        try:
            best_lag = int(stats_row["best_lag"])
            coeffs = ast.literal_eval(stats_row["best_model_parameters"])
            coeffs = np.asarray(coeffs, dtype=float)
        except Exception:
            return None

        if effect_metric == "direct":
            val = coeffs[-best_lag]
        elif effect_metric == "mean":
            val = float(np.nanmean(coeffs[-best_lag:]))
        elif effect_metric == "rms":
            val = np.sign(np.nanmean(coeffs[-best_lag:])) * np.sqrt(np.nanmean(coeffs[-best_lag:] ** 2))
        elif effect_metric == "max":
            subset = coeffs[-best_lag:]
            val = subset[np.argmax(np.abs(subset))]
        else:
            raise ValueError(f"Unsupported effect_metric: {effect_metric}")
        return val

    # ----------------------------- Load & Aggregate ----------------------------- #
    effect_dict: dict[str, dict[str, float]] = {}  # trial → {predictor → effect}

    for path in csv_paths:
        path = Path(path)
        trial_name = path.stem
        df = pd.read_csv(path, index_col=0)

        row_dict = {}
        for predictor, stats in df.iterrows():
            try:
                p = float(stats["p_value"])
                if p >= alpha:
                    continue
                val = extract_effect(stats)
                if val is None:
                    continue
                if expected_sign is not None and np.sign(val) != expected_sign:
                    val = -val
                row_dict[predictor] = val
            except Exception as e:
                print(f"[WARN] Skipping {predictor} in {trial_name}: {e}")
                continue

        if row_dict:
            effect_dict[trial_name] = row_dict

    if not effect_dict:
        print("[WARN] No significant causal effects found.")
        return None

    # ----------------------------- Construct Heatmap Matrix ----------------------------- #
    effect_df = pd.DataFrame.from_dict(effect_dict, orient="index").T  # predictors × trials
    effect_df = effect_df.replace([np.inf, -np.inf], np.nan)
    effect_df = effect_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    effect_df = effect_df.fillna(0.0)  # 或改为 df.fillna(df.median(axis=1), axis=0)

    if effect_df.empty:
        print("[WARN] Effect matrix is empty after filtering.")
        return None

    n_rows, n_cols = effect_df.shape
    vmax = np.nanpercentile(np.abs(effect_df.values), 99)

    if figsize is None:
        figsize = (max(12, 0.08 * n_cols), max(8, 0.25 * n_rows))

    sns.set_theme(style="white")
    cg = sns.clustermap(
        effect_df,
        row_cluster=row_cluster,
        col_cluster=False,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.3,
        figsize=figsize,
        cbar_kws={"label": f"Causal effect ({effect_metric})"},
    )
    ax = cg.ax_heatmap

    _set_axis_labels(ax, effect_df, n_rows, n_cols)

    if not title:
        title = (
            f"Causal effect heatmap ({effect_metric}, α={alpha})"
        )
    ax.set_title(title, pad=40)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        return None

    return ax


def _set_axis_labels(ax: plt.Axes, df: pd.DataFrame, n_rows: int, n_cols: int) -> None:
    if n_cols <= 80:
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_xticklabels(df.columns, rotation=90, fontsize=max(4, 9 - int(np.log10(n_cols))))
    elif n_cols <= 200:
        step = int(np.ceil(n_cols / 80))
        ticks = np.arange(0, n_cols, step) + 0.5
        ax.set_xticks(ticks)
        ax.set_xticklabels(df.columns[::step], rotation=90, fontsize=4)
    else:
        ax.set_xticks([])
        ax.set_xlabel(f"Trials (n={n_cols})", fontsize=10)

    if n_rows <= 60:
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_yticklabels(df.index, fontsize=max(4, 9 - int(np.log10(n_rows))))
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"Predictors (n={n_rows})", fontsize=10)