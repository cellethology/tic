# tic/features/fourier.py
"""
Fourier-based feature extractor for TIC pipeline.

This module defines:
  - A function to compute low-frequency Fourier descriptor magnitudes
  - A FeatureExtractor implementation that integrates with TIC's registry
    to extract Fourier descriptors from cell boundary polygons.
"""

from typing import Sequence, List
import numpy as np
from anndata import AnnData

from ..base import FeatureExtractor
from ..registry import register

# -------------------------------------------------------------------
# Boundary normalization
# -------------------------------------------------------------------
def normalize_boundary(
    pts: np.ndarray,
    normalize_area: bool = True
) -> np.ndarray:
    """Center boundary at origin and optionally normalize polygon area."""
    pts = pts - pts.mean(axis=0)
    if normalize_area:
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area > 0:
            pts = pts / np.sqrt(area)
    return pts

# -------------------------------------------------------------------
# Fourier reconstruction & descriptors
# -------------------------------------------------------------------
def reconstruct_from_fourier(
    z: np.ndarray,
    k: int
) -> np.ndarray:
    """Keep k low frequencies and perform inverse FFT to reconstruct shape."""
    Z = np.fft.fft(z)
    if k < len(Z) // 2:
        Z[k+1 : -k] = 0
    recon = np.fft.ifft(Z)
    return np.vstack([recon.real, recon.imag]).T

def compute_fourier_descriptors(
    points: np.ndarray,
    n_descriptors: int = 10,
    normalize_area: bool = True,
) -> np.ndarray:
    """
    Compute low-frequency Fourier descriptor magnitudes from 2D boundary polygon.
    Excludes the DC component; returns abs(Z[1:n_descriptors+1]).
    """
    if points.shape[0] < 3:
        return np.full(n_descriptors, np.nan)

    # ensure closed
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    pts = normalize_boundary(points, normalize_area)
    z = pts[:, 0] + 1j * pts[:, 1]
    Z = np.fft.fft(z)
    return np.abs(Z[1 : n_descriptors + 1])

# -------------------------------------------------------------------
# Error metrics
# -------------------------------------------------------------------
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return np.mean(np.sum(np.abs(a - b), axis=1))

_METRICS = {
    "rmse": rmse,
    "mae": mae,
}

@register
class FourierFeatureExtractor(FeatureExtractor):
    """
    Extract low-frequency Fourier descriptors from the boundary polygon of the centre cell.

    Each descriptor is the magnitude of the k-th Fourier coefficient (excluding DC term).
    Shape is normalized by centroid alignment and optional area normalization.

    Attributes
    ----------
    _n_descriptors : int
        Number of Fourier descriptors to compute.
    _normalize_area : bool
        Whether to normalize polygon area to 1 before computing descriptors.
    """

    name = "fourier_features"

    def __init__(
        self,
        n_descriptors: int = 10,
        normalize_area: bool = True,
    ):
        """
        Parameters
        ----------
        n_descriptors : int, default 10
            Number of low-frequency descriptors to extract.
        normalize_area : bool, default True
            Whether to normalize boundary area to unity.
        """
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
        """
        Compute Fourier descriptors for a specified centre cell.

        Parameters
        ----------
        adata : AnnData
            Annotated data containing 'cell_boundaries' in .uns.
        centre_idx : int
            Index of the centre cell in adata.obs_names.
        neighbour_idx : Sequence[int]
            Ignored—maintained for API compatibility.

        Returns
        -------
        descriptors : numpy.ndarray, shape (n_descriptors,)
            Magnitudes of the first n_descriptors Fourier coefficients.
        """
        cell_id = adata.obs_names[centre_idx]
        boundary_df = adata.uns["cell_boundaries"]["cell"]
        points = boundary_df[
            boundary_df["cell_id"] == cell_id
        ][["vertex_x", "vertex_y"]].values

        return compute_fourier_descriptors(
            points,
            n_descriptors=self._n_descriptors,
            normalize_area=self._normalize_area,
        )

    @property
    def n_features(self) -> int:
        """
        Number of features produced by this extractor (equals n_descriptors).
        """
        return self._n_descriptors

    def feature_names(self, adata: AnnData) -> List[str]:
        """
        Generate feature names for each Fourier descriptor.

        Returns
        -------
        names : List[str]
            List of strings like 'fourier:1', 'fourier:2', ..., up to n_descriptors.
        """
        return [f"fourier:{k+1}" for k in range(self._n_descriptors)]
