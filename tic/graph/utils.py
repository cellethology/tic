"""Helper utilities shared across ``tic.graph`` modules."""
from __future__ import annotations

from anndata import AnnData
from matplotlib import pyplot as plt
import numpy as np


from ..constant import DEFAULT_KEY


def get_connectivities_key(custom_key: str | None = None) -> str:
    """Return the connectivities key to be used in ``adata.obsp``.

    Parameters
    ----------
    custom_key
        If provided, overrides the default ``"connectivities"``.
    """
    return custom_key or DEFAULT_KEY.get('connectivities_key')

def estimate_radius(
    adata: AnnData,
    target_microenv_size: int = 30,
    n_samples: int = 1000,
    visualize: bool = False,
) -> float:
    """
    Estimate a spatial radius that includes approximately `target_microenv_size` cells
    on average across random locations, and optionally visualize the distribution.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix with `.obsm['spatial']` coordinates.
    target_microenv_size : int
        The target number of cells in a microenvironment.
    n_samples : int, default=1000
        Number of random cells to sample when estimating the average radius.
    visualize : bool
        Whether to plot the distribution of radii.

    Returns
    -------
    float
        Estimated radius in spatial units.
    """
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    n_cells = coords.shape[0]

    if target_microenv_size >= n_cells:
        raise ValueError("Target microenvironment size exceeds total number of cells.")

    sampled_indices = np.random.choice(n_cells, size=min(n_samples, n_cells), replace=False)
    distances = []

    for idx in sampled_indices:
        dists = np.linalg.norm(coords - coords[idx], axis=1)
        sorted_dists = np.sort(dists)
        radius = sorted_dists[target_microenv_size]  # include self
        distances.append(radius)

    distances = np.array(distances)
    estimated_radius = float(np.median(distances))

    if visualize:
        plt.figure(figsize=(6, 4))
        plt.hist(distances, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        plt.axvline(estimated_radius, color="red", linestyle="--", label=f"Median radius ≈ {estimated_radius:.2f}")
        plt.xlabel("Radius (distance)")
        plt.ylabel("Frequency")
        plt.title(f"Estimated Radius for {target_microenv_size} Neighbors")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return estimated_radius