from __future__ import annotations

from typing import Sequence, List

import numpy as np
from anndata import AnnData

from tic.features.base import FeatureExtractor
from tic.features.registry import register


def compute_fourier_descriptors(
    points: np.ndarray,
    n_descriptors: int = 10,
    normalize_area: bool = True,
) -> np.ndarray:
    """
    Compute low-frequency Fourier descriptors from 2D boundary polygon.

    Parameters
    ----------
    points : np.ndarray
        Nx2 array of (x, y) coordinates.
    n_descriptors : int
        Number of Fourier coefficients to return.
    normalize_area : bool
        Whether to normalize polygon area to 1.

    Returns
    -------
    fd : np.ndarray
        Real-valued vector of descriptor magnitudes.
    """
    if points.shape[0] < 3:
        return np.full(n_descriptors, np.nan)

    # Ensure closed polygon
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    # Center the shape
    centroid = points.mean(axis=0)
    points -= centroid

    # Normalize area to 1 if requested
    if normalize_area:
        x, y = points[:, 0], points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area > 0:
            points /= np.sqrt(area)

    # Convert to complex sequence
    z = points[:, 0] + 1j * points[:, 1]
    Z = np.fft.fft(z)

    # Magnitude of first `n` frequencies (excluding DC)
    return np.abs(Z[1 : n_descriptors + 1])


@register
class FourierFeatureExtractor(FeatureExtractor):
    """
    Extract low-frequency Fourier descriptors from the boundary polygon of the centre cell.
    The shape is normalized by centroid alignment and optional area normalization.
    """

    name = "fourier_features"

    def __init__(self, n_descriptors: int = 10, normalize_area: bool = True):
        super().__init__()
        self._n_descriptors = n_descriptors
        self._normalize_area = normalize_area

    def transform(
        self,
        adata: AnnData,
        *,
        centre_idx: int,
        neighbour_idx: Sequence[int],
    ) -> np.ndarray:
        cell_id = adata.obs_names[centre_idx]
        boundary_df = adata.uns["cell_boundaries"]["cell"]
        points = boundary_df[boundary_df["cell_id"] == cell_id][["vertex_x", "vertex_y"]].values

        return compute_fourier_descriptors(
            points,
            n_descriptors=self._n_descriptors,
            normalize_area=self._normalize_area,
        )

    @property
    def n_features(self) -> int:
        return self._n_descriptors

    def feature_names(self, adata: AnnData) -> List[str]:
        return [f"fourier:{k+1}" for k in range(self._n_descriptors)]