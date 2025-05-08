# file: tic/pipeline/pseudotime.py
"""High-level, order-flexible pseudotime pipeline."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence
import numpy as np
import pandas as pd
from anndata import AnnData

from tic.constant import DEFAULT_KEY

from ..config import PseudotimeConfig
from ..metrics.api import calculate_monotonicity, calculate_trend
from ..pseudotime.pp.clustering import Clustering
from ..pseudotime.pp.dimensionality import DimensionalityReduction
from ..pseudotime.tl.slingshot import SlingshotMethod
from ..utils.logging import get_logger

log = get_logger(__name__)


class _PipelineStep(Protocol):
    """Every concrete step mutates the input *in‑place*."""

    def __call__(self, adata: AnnData) -> None:  # noqa: D401 – protocol method
        ...


class _DRStep:  # noqa: D401 – step class
    """Dimensionality‑reduction (PCA / UMAP / etc.)."""

    def __init__(self, cfg: PseudotimeConfig) -> None:  # noqa: D401
        self._cfg = cfg
        self._dr = DimensionalityReduction(
            method=cfg.dr_method,
            n_components=cfg.n_components,
            random_state=cfg.random_state,
        )

    # ------------------------------------------------------------------
    def __call__(self, adata: AnnData) -> None:  # noqa: D401 – step protocol
        X = adata.obsm.get(self._cfg.rep_key, adata.X)
        adata.obsm["rp_reduced"] = self._dr.fit_transform(X)


class _ClusterStep:
    """Cluster cells in the (optionally) reduced space."""

    def __init__(self, cfg: PseudotimeConfig) -> None:  # noqa: D401
        self._cfg = cfg
        self._cluster = Clustering(
            method=cfg.cluster_method,
            n_clusters=cfg.n_clusters,
            random_state=cfg.random_state,
        )

    # ------------------------------------------------------------------
    def __call__(self, adata: AnnData) -> None:  # noqa: D401
        X = adata.obsm.get("rp_reduced", adata.obsm.get(self._cfg.rep_key, adata.X))
        adata.obs["cluster"] = self._cluster.fit_predict(X)


class _PseudotimeStep:
    """Run Slingshot and attach pseudotime to ``adata.obs``."""

    def __init__(self, cfg: PseudotimeConfig) -> None:  # noqa: D401
        self._engine = SlingshotMethod(start_node=cfg.start_node, epochs=cfg.epochs)
        self._outdir = cfg.output_dir

    # ------------------------------------------------------------------
    def __call__(self, adata: AnnData) -> None:  # noqa: D401
        emb = adata.obsm.get("rp_reduced")
        labels = adata.obs["cluster"].to_numpy()

        # Need at least two clusters and two cells, otherwise Slingshot is moot.
        if emb is None or emb.shape[0] < 2 or len(np.unique(labels)) < 2:
            log.warning(
                "Skipping Slingshot: require ≥2 cells *and* ≥2 clusters."
            )
            adata.obs["pseudotime"] = np.zeros(adata.n_obs)
            return

        try:
            pseudo = self._engine.fit_predict(
                emb,
                labels,
                output_dir=(str(self._outdir) if self._outdir else None),
            )
        except Exception as exc:  # broad catch → guaranteed not to crash pipeline
            log.error("Slingshot failed – setting pseudotime to zero: %r", exc)
            pseudo = np.zeros(adata.n_obs)

        adata.obs["pseudotime"] = pseudo


class _MetricStep:
    """Compute per-gene monotonicity & trend statistics."""

    def __init__(self, cfg: PseudotimeConfig) -> None:  # noqa: D401
        self._mt_method = cfg.monotonicity_method
        self._tr_method = cfg.trend_method

    # ------------------------------------------------------------------
    def __call__(self, adata: AnnData) -> None:
        pt = adata.obs["pseudotime"].to_numpy()
        expr = np.asarray(adata.X)
        genes = list(adata.var_names)

        records: list[dict] = []
        for idx, gene in enumerate(genes):
            y = expr[:, idx]
            mono = calculate_monotonicity(y, pt, method=self._mt_method)
            tr = calculate_trend(y, pt, method=self._tr_method)

            entry = {"biomarker": gene}
            entry.update({f"mono_{k}": v for k, v in mono.items()})
            entry.update({f"trend_{k}": v for k, v in tr.items()})
            records.append(entry)

        adata.uns["pseudotime_metrics"] = pd.DataFrame.from_records(records)


# Mapping from step‑name → implementation
_STEP_FACTORY: Mapping[str, type[_PipelineStep]] = {
    "dr": _DRStep,
    "cluster": _ClusterStep,
    "slingshot": _PseudotimeStep,
    "metrics": _MetricStep,
}

# -----------------------------------------------------------------------------
# Public pipeline orchestrator
# -----------------------------------------------------------------------------


class PseudotimePipeline:
    """Run a *configurable* sequence of pseudotime analysis steps.

    Parameters
    ----------
    cfg
        A :class:`PseudotimeConfig` instance.
    steps
        Optional explicit order (subset / permutation of {"dr", "cluster",
        "slingshot", "metrics"}).  If *None*, falls back to
        ``cfg.step_order`` or the default `("cluster", "dr", "slingshot", "metrics")`.
    """

    def __init__(self, cfg: PseudotimeConfig, *, steps: Sequence[str] | None = None) -> None:
        self._cfg = cfg
        order: Sequence[str] = steps or cfg.step_order or (
            "cluster", "dr", "slingshot", "metrics"
        )

        # Validate & instantiate concrete step objects
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

        ad.uns[DEFAULT_KEY.get('pipeline_config')] = self._cfg.to_dict()
        return ad