# file: tic/pipeline/pseudotime.py
"""High-level, configurable pseudotime pipeline.

Standard workflow
-----------------
(1) Standardization         -> stores  ``.obsm['rp_scaled']``
(2) Dimensionality reduction -> stores  ``.obsm['rp_reduced']``
(3) Clustering               -> writes  ``.obs['cluster']``
(4) Slingshot pseudotime     -> writes  ``.obs['pseudotime']``
(5) Metric computation       -> writes  ``.uns['pseudotime_metrics']``
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler

from ..constant import DEFAULT_KEY
from ..config import PseudotimeConfig
from ..metrics.api import calculate_monotonicity, calculate_trend
from ..pseudotime.pp.clustering import Clustering
from ..pseudotime.pp.dimensionality import DimensionalityReduction
from ..pseudotime.tl.slingshot import SlingshotMethod
from ..utils.logging import get_logger

log = get_logger(__name__)


# -----------------------------------------------------------------------------#
# Step protocol
# -----------------------------------------------------------------------------#
class _PipelineStep(Protocol):
    """Every concrete step mutates the input *in-place*."""

    def __call__(self, adata: AnnData) -> None:  # noqa: D401
        ...


# -----------------------------------------------------------------------------#
# STEP 1 ── Standardization
# -----------------------------------------------------------------------------#
class _StandardizeStep:
    """Z-score each feature (gene) across cells."""

    _TARGET_KEY = "rp_scaled"

    def __init__(self, cfg: PseudotimeConfig) -> None:  # noqa: D401
        self._cfg = cfg
        # Keep zero-center for dense; drop mean for sparse
        self._scaler_dense = StandardScaler(with_mean=True, with_std=True)
        self._scaler_sparse = StandardScaler(with_mean=False, with_std=True)

    # ------------------------------------------------------------------
    def __call__(self, adata: AnnData) -> None:  # noqa: D401
        X_raw = adata.obsm.get(self._cfg.rep_key, adata.X)
        scaler = self._scaler_sparse if issparse(X_raw) else self._scaler_dense
        adata.obsm[self._TARGET_KEY] = scaler.fit_transform(X_raw)
        log.debug("Standardized data stored in `.obsm['%s']`", self._TARGET_KEY)


# -----------------------------------------------------------------------------#
# STEP 2 ── Dimensionality reduction
# -----------------------------------------------------------------------------#
class _DRStep:
    """Dimensionality-reduction (PCA / UMAP / etc.)."""

    _TARGET_KEY = "rp_reduced"

    def __init__(self, cfg: PseudotimeConfig) -> None:  # noqa: D401
        self._cfg = cfg
        self._dr = DimensionalityReduction(
            method=cfg.dr_method,
            n_components=cfg.n_components,
            random_state=cfg.random_state,
        )

    # ------------------------------------------------------------------
    def __call__(self, adata: AnnData) -> None:  # noqa: D401
        if _StandardizeStep._TARGET_KEY in adata.obsm:
            X = adata.obsm[_StandardizeStep._TARGET_KEY]
        elif self._cfg.rep_key in adata.obsm:
            X = adata.obsm[self._cfg.rep_key]
        else:
            X = adata.X

        adata.obsm[self._TARGET_KEY] = self._dr.fit_transform(X)
        log.debug("Reduced representation stored in `.obsm['%s']`", self._TARGET_KEY)


# -----------------------------------------------------------------------------#
# STEP 3 ── Clustering
# -----------------------------------------------------------------------------#
class _ClusterStep:
    """Cluster cells in the reduced (2-D) space."""

    def __init__(self, cfg: PseudotimeConfig) -> None:  # noqa: D401
        self._cfg = cfg
        self._cluster = Clustering(
            method=cfg.cluster_method,
            n_clusters=cfg.n_clusters,
            random_state=cfg.random_state,
        )

    # ------------------------------------------------------------------
    def __call__(self, adata: AnnData) -> None:  # noqa: D401
        X = adata.obsm.get(_DRStep._TARGET_KEY)  # must exist after DR
        if X is None:
            raise RuntimeError(
                "Clustering step requires reduced embedding; "
                "make sure 'dr' precedes 'cluster'."
            )
        adata.obs["cluster"] = self._cluster.fit_predict(X)
        log.debug("Cluster labels stored in `.obs['cluster']`")


# -----------------------------------------------------------------------------#
# STEP 4 ── Pseudotime (Slingshot)
# -----------------------------------------------------------------------------#
class _PseudotimeStep:
    """Run Slingshot and attach pseudotime to ``adata.obs``."""

    def __init__(self, cfg: PseudotimeConfig) -> None:  # noqa: D401
        self._engine = SlingshotMethod(start_node=cfg.start_node, epochs=cfg.epochs)
        self._outdir = cfg.output_dir

    # ------------------------------------------------------------------
    def __call__(self, adata: AnnData) -> None:  # noqa: D401
        emb = adata.obsm.get(_DRStep._TARGET_KEY)
        labels = adata.obs.get("cluster")

        if emb is None or labels is None:
            log.warning("Skipping Slingshot: embedding or clusters missing.")
            adata.obs["pseudotime"] = np.zeros(adata.n_obs)
            return

        labels_np = labels.to_numpy()
        if emb.shape[0] < 2 or len(np.unique(labels_np)) < 2:
            log.warning("Skipping Slingshot: need ≥2 cells *and* ≥2 clusters.")
            adata.obs["pseudotime"] = np.zeros(adata.n_obs)
            return

        try:
            pseudo = self._engine.fit_predict(
                emb,
                labels_np,
                output_dir=(str(self._outdir) if self._outdir else None),
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Slingshot failed – setting pseudotime to zero: %r", exc)
            pseudo = np.zeros(adata.n_obs)

        adata.obs["pseudotime"] = pseudo
        log.debug("Pseudotime vector stored in `.obs['pseudotime']`")


# -----------------------------------------------------------------------------#
# STEP 5 ── Metrics
# -----------------------------------------------------------------------------#
class _MetricStep:
    """Compute gene-wise monotonicity & trend statistics."""

    def __init__(self, cfg: PseudotimeConfig) -> None:  # noqa: D401
        self._mt_method = cfg.monotonicity_method
        self._tr_method = cfg.trend_method

    # ------------------------------------------------------------------
    def __call__(self, adata: AnnData) -> None:  # noqa: D401
        pt = adata.obs.get("pseudotime")
        if pt is None:
            raise RuntimeError("Pseudotime not found; run 'slingshot' first.")

        pt_values = pt.to_numpy()
        expr = np.asarray(adata.X)
        genes = list(adata.var_names)

        records: list[dict[str, Any]] = []
        for idx, gene in enumerate(genes):
            y = expr[:, idx]
            mono = calculate_monotonicity(y, pt_values, method=self._mt_method)
            tr = calculate_trend(y, pt_values, method=self._tr_method)

            entry: dict[str, Any] = {"biomarker": gene}
            entry.update({f"mono_{k}": v for k, v in mono.items()})
            entry.update({f"trend_{k}": v for k, v in tr.items()})
            records.append(entry)

        adata.uns["pseudotime_metrics"] = pd.DataFrame.from_records(records)
        log.debug("Metric table stored in `.uns['pseudotime_metrics']`")


# -----------------------------------------------------------------------------#
# Factory & Pipeline
# -----------------------------------------------------------------------------#
_STEP_FACTORY: Mapping[str, type[_PipelineStep]] = {
    "standardize": _StandardizeStep,
    "dr": _DRStep,
    "cluster": _ClusterStep,
    "slingshot": _PseudotimeStep,
    "metrics": _MetricStep,
}


class PseudotimePipeline:
    """Run a configurable sequence of pseudotime analysis steps."""

    DEFAULT_ORDER: tuple[str, ...] = (
        "standardize",
        "dr",
        "cluster",
        "slingshot",
        "metrics",
    )

    def __init__(self, cfg: PseudotimeConfig, *, steps: Sequence[str] | None = None) -> None:
        self._cfg = cfg
        order: Sequence[str] = steps or cfg.step_order or self.DEFAULT_ORDER

        self._steps: list[_PipelineStep] = []
        for name in order:
            if name not in _STEP_FACTORY:
                raise ValueError(f"Unknown step '{name}'. Valid: {list(_STEP_FACTORY)}")
            self._steps.append(_STEP_FACTORY[name](cfg))

    # ------------------------------------------------------------------
    def run(self, adata: AnnData, *, copy: bool = True) -> AnnData:
        """Execute the pipeline and return the mutated (or copied) `AnnData`."""
        ad = adata.copy() if copy else adata
        log.info("Pseudotime pipeline order: %s", [s.__class__.__name__ for s in self._steps])

        for step in self._steps:
            step(ad)

        ad.uns[DEFAULT_KEY.get("pipeline_config")] = {
            **self._cfg.to_dict(),
            "pipeline_version": "v2-standardize-first",
        }
        return ad