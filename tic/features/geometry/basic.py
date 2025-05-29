from __future__ import annotations

from typing import Sequence, List

import numpy as np
from anndata import AnnData

from ..base import FeatureExtractor
from ..registry import register


@register
class GeometryFeatureExtractor(FeatureExtractor):
    """
    Extract scalar geometry features (area, perimeter, aspect_ratio, compactness, centroid)
    from the boundary polygon of the centre cell.
    Requires `adata.uns['cell_boundaries']['cell']` to be available.

    Returned features:
    - area: float, the area of the polygon
    - perimeter: float, the perimeter of the polygon
    - aspect_ratio: float, the aspect ratio of the polygon
    - compactness: float, the compactness of the polygon
    - centroid_x: float, the x-coordinate of the centroid
    - centroid_y: float, the y-coordinate of the centroid
    """

    name = "geometry_basic"

    def __init__(self):
        super().__init__()

    def transform(
        self,
        adata: AnnData,
        *,
        centre_idx: int,
        neighbour_idx: Sequence[int],
    ) -> np.ndarray:
        assert "cell_boundaries" in adata.uns, "cell_boundaries not found in adata.uns"
        assert "cell" in adata.uns["cell_boundaries"], "cell not found in cell_boundaries"
        boundaries = adata.uns["cell_boundaries"]["cell"]
        cell_id = adata.obs_names[centre_idx]

        polygon = boundaries[boundaries["cell_id"] == cell_id][["vertex_x", "vertex_y"]].values

        if polygon.shape[0] < 3:
            return np.full(self.n_features, np.nan)  # Invalid polygon

        # Ensure polygon is closed
        if not np.allclose(polygon[0], polygon[-1]):
            polygon = np.vstack([polygon, polygon[0]])

        x, y = polygon[:, 0], polygon[:, 1]

        # Area (Shoelace formula)
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        # Perimeter
        dx = np.diff(x)
        dy = np.diff(y)
        perimeter = np.sum(np.sqrt(dx ** 2 + dy ** 2))

        # Aspect ratio
        width = np.max(x) - np.min(x)
        height = np.max(y) - np.min(y)
        aspect_ratio = width / height if height != 0 else np.nan

        # Compactness
        compactness = (perimeter ** 2) / area if area != 0 else np.nan

        # Centroid
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)

        return np.array([
            area, perimeter, aspect_ratio, compactness, centroid_x, centroid_y
        ], dtype=float)

    @property
    def n_features(self) -> int:
        return 6

    def feature_names(self, adata: AnnData) -> List[str]:
        return [
            "area",
            "perimeter",
            "aspect_ratio",
            "compactness",
            "centroid_x",
            "centroid_y"
        ]
