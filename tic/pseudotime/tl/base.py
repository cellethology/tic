"""Abstract base class for pseudotime methods."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["PseudotimeMethod"]


class PseudotimeMethod(ABC):
    """Abstract interface every pseudotime algorithm must implement."""

    @abstractmethod
    def fit_predict(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        *,
        output_dir: str | None = None,
    ) -> np.ndarray:  # noqa: D401
        """Return pseudotime values for each sample."""
