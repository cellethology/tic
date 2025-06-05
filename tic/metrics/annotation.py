"""
tic.metrics.annotation
======================

Functions
---------
- evaluate_cell_type_mapping:
    Compute optimal label alignment between true and predicted cell‐type annotations
    (when both sets of labels differ in name and/or cardinality), and return standard
    classification metrics.

- plot_aligned_confusion_matrix:
    Given true labels and mapped predicted labels, draw a publication‐ready confusion
    matrix with annotations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, List
from anndata import AnnData
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    adjusted_rand_score,
)
from scipy.optimize import linear_sum_assignment


def evaluate_cell_type_mapping(
    adata: AnnData,
    true_key: str = "cell_type",
    pred_key: str = "pred_cell_type",
    mapped_key: str = "pred_cell_type_mapped",
    return_mapping: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Compute an optimal one‐to‐one alignment between predicted and true labels
    (using the Hungarian algorithm), store the aligned predictions in
    `adata.obs[mapped_key]`, and return mapping + metrics.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with `.obs[true_key]` and `.obs[pred_key]` columns.
    true_key : str
        Key in `.obs` corresponding to the ground‐truth labels.
    pred_key : str
        Key in `.obs` corresponding to the raw predicted labels.
    mapped_key : str
        Key under which the aligned (mapped) predicted labels will be stored in `.obs`.
    return_mapping : bool
        If True, return the mapping dictionary (pred_label → true_label). Otherwise mapping is None.
    verbose : bool
        If True, print the mapping dictionary, classification report, and ARI to stdout.

    Returns
    -------
    results : dict
        A dictionary with keys:
          - "mapping": (optional) dict from each original pred_label → aligned true_label
          - "classification_report": the sklearn classification report (str) comparing
            y_true vs. y_pred_mapped
          - "adjusted_rand_index": float ARI between y_true and y_pred (raw)
    """
    # Extract true and predicted labels as string arrays
    y_true = adata.obs[true_key].astype(str).values
    y_pred = adata.obs[pred_key].astype(str).values

    # Determine the unique label sets
    true_labels: List[str] = sorted(set(y_true))
    pred_labels: List[str] = sorted(set(y_pred))

    # Build a contingency table (true_labels × pred_labels)
    contingency_df = pd.crosstab(
        pd.Series(y_true, name="true"),
        pd.Series(y_pred, name="pred"),
        dropna=False,
    )
    # Reindex rows and columns to ensure all labels appear
    contingency_df = contingency_df.reindex(
        index=true_labels, columns=pred_labels, fill_value=0
    )

    # Apply Hungarian algorithm to maximize total matches
    # (we minimize the negative of the contingency table)
    cost_matrix = -contingency_df.values  # shape = (n_true, n_pred)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build the mapping: pred_label -> true_label
    mapping: Dict[str, str] = {
        pred_labels[col]: true_labels[row]
        for row, col in zip(row_ind.tolist(), col_ind.tolist())
    }

    # Map raw predictions to aligned true labels; unmapped preds get "Unmapped"
    y_pred_mapped = (
        pd.Series(y_pred).map(mapping).fillna("Unmapped").astype(str).values
    )
    adata.obs[mapped_key] = y_pred_mapped

    # Generate a classification report
    cls_report = classification_report(
        y_true, y_pred_mapped, zero_division=0
    )

    # Compute Adjusted Rand Index on the raw (unaligned) labels
    ari_raw = adjusted_rand_score(y_true, y_pred)

    if verbose:
        print("=== Label Mapping (pred → true) ===")
        for p_label, t_label in mapping.items():
            print(f"{p_label:30s} → {t_label}")
        print("\n=== Classification Report (after alignment) ===")
        print(cls_report)
        print(f"Adjusted Rand Index (raw): {ari_raw:.4f}")

    return {
        "mapping": mapping if return_mapping else None,
        "classification_report": cls_report,
        "adjusted_rand_index": ari_raw,
    }


def plot_aligned_confusion_matrix(
    adata: AnnData,
    true_key: str = "cell_type",
    mapped_key: str = "pred_cell_type_mapped",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    annotate: bool = True,
    fmt: str = "d",
    xlabel: str = "Predicted (aligned)",
    ylabel: str = "Ground Truth",
    title: str = "Aligned Confusion Matrix",
    xtick_rotation: int = 45,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a nicely formatted confusion matrix given the ground‐truth labels
    and the aligned (mapped) predicted labels.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with `.obs[true_key]` and `.obs[mapped_key]` columns.
    true_key : str
        Key in `.obs` for the ground‐truth labels.
    mapped_key : str
        Key in `.obs` for the aligned predicted labels (output of evaluate_cell_type_mapping).
    figsize : tuple
        Figure size in inches, e.g. (width, height).
    cmap : str
        Matplotlib colormap to use for the heatmap.
    annotate : bool
        If True, annotate each cell with its integer count.
    fmt : str
        Format string for annotation (e.g. "d" for integers, ".1f" for floats).
    xlabel : str
        Label for the x‐axis.
    ylabel : str
        Label for the y‐axis.
    title : str
        Plot title.
    xtick_rotation : int
        Rotation angle for x‐tick labels.
    save_path : str or None
        If provided, save the figure to this path (e.g. "output/cm.png").
    """
    # Extract true + mapped predictions
    y_true = adata.obs[true_key].astype(str).values
    y_pred_mapped = adata.obs[mapped_key].astype(str).values

    # Determine the set of true labels (including "Unmapped" if present)
    unique_true_labels = sorted(set(y_true))

    # Build confusion matrix aligned on true_labels
    cm = confusion_matrix(
        y_true, y_pred_mapped, labels=unique_true_labels
    )
    cm_df = pd.DataFrame(cm, index=unique_true_labels, columns=unique_true_labels)

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_df,
        annot=annotate,
        fmt=fmt,
        cmap=cmap,
        cbar=True,
        square=False,
        linewidths=0.5,
        linecolor="gray",
    )
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(
        rotation=xtick_rotation,
        ha="right",
        fontsize=10,
    )
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()