# file: tic/wrappers/causal.py
"""
High‑level wrapper that converts pseudo‑time + neighbourhood features
into a tidy DataFrame → runs causal inference → stores results in `adata.uns`.
"""

from __future__ import annotations
from typing import Any, Dict, Literal, Sequence, List, Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

from ..constant import DEFAULT_KEY
from ..causal.factory import CausalMethodFactory
from ..plotting import (
    plot_causal_heatmap,
    plot_causal_bar,
    plot_causal_volcano,
)


class CausalWrapper:  # pylint: disable=too-few-public-methods
    """
    One‑stop causal inference + plotting interface.

    Parameters
    ----------
    outcome
        Biomarker name (must be in ``adata.var_names``) used as *Y*.
    feature_key
        Matrix of *all* potential predictors (defaults to ``"X_predictors"``).
    include_extractors
        Only use columns whose extractor prefix (before the *first* ``":"``)
        appears in this list.  E.g. ``["celltype_gene_count"]``.
        If ``None`` → keep every column in ``feature_key``.
    method
        Causal method string recognised by :pyclass:`tic.causal.factory`.
    bins
        Number of pseudotime bins.  ``None`` or ``<=1`` → no binning.
    method_kwargs
        Extra kwargs forwarded to the causal method constructor.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        outcome: str,
        feature_key: str = "X_predictors",
        include_extractors: Optional[Sequence[str]] = ("celltype_gene_count",),
        method: str = "granger_causality",
        bins: int | None = 100,
        method_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.outcome = outcome
        self.feature_key = feature_key
        self.include_extractors: Optional[tuple[str, ...]] = (
            tuple(include_extractors) if include_extractors is not None else None
        )
        self.method = method
        self.bins = bins
        self.method_kwargs = method_kwargs or {}

        self._results: Dict[str, Dict[str, Any]] | None = None  # cache

    # --------------------------------------------------------------------- public
    def fit(self, adata: AnnData) -> Dict[str, Dict[str, Any]]:
        """Run causal inference for every selected predictor."""
        df = self._prepare_dataframe(adata)
        predictors = [c for c in df.columns if c not in ("time", "Y")]

        from ..causal.causal_input import CausalInput

        results: Dict[str, Dict[str, Any]] = {}
        for pred in predictors:
            ci = CausalInput(data=df[["Y", pred]].dropna(),
                             treatment_col=pred,
                             outcome_col="Y")
            method = CausalMethodFactory.get_method(self.method, **self.method_kwargs)
            method.fit(ci)
            results[pred] = method.estimate_effect(ci)

        self._results = results
        adata.uns[DEFAULT_KEY.get("causal_results")] = results
        return adata

    def plot(self, adata: AnnData, kind: str | Literal["heatmap", "bar", "volcano"] = "bar", **kwargs) -> None:
        """
        Plot the causal results.

        Parameters
        ----------
        adata
            Annotated data object.
        kind
            for more details, see :func:`plot_causal_heatmap`, :func:`plot_causal_bar`, :func:`plot_causal_volcano`. at tic.plotting.casual
        """
        _map = {"heatmap": plot_causal_heatmap,
                "bar": plot_causal_bar,
                "volcano": plot_causal_volcano}
        if kind not in _map:
            raise ValueError(f"Unknown plot kind '{kind}'.")
        _map[kind](adata, **kwargs)

    # ------------------------------------------------------------------ helpers
    def _prepare_dataframe(self, adata: AnnData) -> pd.DataFrame:
        # 1) -------- sanity
        pt_key = DEFAULT_KEY.get("pseudotime")
        if pt_key not in adata.obs:
            raise KeyError(f"Missing pseudo‑time in .obs['{pt_key}'].")
        if self.outcome not in adata.var_names:
            raise ValueError(f"Outcome '{self.outcome}' not in adata.var_names.")
        if self.feature_key not in adata.obsm:
            raise KeyError(f"Feature matrix .obsm['{self.feature_key}'] not found.")

        # 2) -------- load matrix & names
        mat = adata.obsm[self.feature_key]
        if issparse(mat):
            mat = mat.toarray()
        mat = np.asarray(mat, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat[:, None]

        if "X_predictors_names" in adata.uns:
            all_names: List[str] = list(adata.uns["X_predictors_names"])
        else:
            # fallback – synthetic column names
            all_names = [f"{self.feature_key}:{i}" for i in range(mat.shape[1])]

        if len(all_names) != mat.shape[1]:
            raise ValueError("Column‑name list length does not match feature matrix.")

        # 3) -------- optional extractor filtering
        if self.include_extractors is not None:
            keep_mask = [
                name.split(":", 1)[0] in self.include_extractors
                for name in all_names
            ]
            if not any(keep_mask):
                raise ValueError("No columns match `include_extractors`.")
            mat = mat[:, keep_mask]
            feat_names = [n for n, keep in zip(all_names, keep_mask) if keep]
        else:
            feat_names = all_names

        # 4) -------- bin pseudo‑time & aggregate
        pt = adata.obs[pt_key].to_numpy()
        y_idx = list(adata.var_names).index(self.outcome)

        if self.bins is None or self.bins <= 1:
            bin_ids = np.zeros_like(pt, dtype=int)
            centres = np.array([float(pt.mean())])
        else:
            edges = np.linspace(pt.min(), pt.max(), self.bins + 1)
            bin_ids = np.clip(np.digitize(pt, edges) - 1, 0, self.bins - 1)
            centres = (edges[:-1] + edges[1:]) / 2

        rows: list[dict[str, float]] = []
        for b in np.unique(bin_ids):
            idx = np.where(bin_ids == b)[0]
            if idx.size == 0:
                continue
            row: dict[str, float] = {
                "time": float(centres[b]),
                "Y": float(adata.X[idx, y_idx].mean()),
            }
            row.update({n: float(mat[idx, j].mean()) for j, n in enumerate(feat_names)})
            rows.append(row)

        df = (pd.DataFrame(rows)
                .sort_values("time")
                .reset_index(drop=True))

        # 5) -------- drop constant predictors
        pred_cols = [c for c in df.columns if c not in ("time", "Y")]
        const = [c for c in pred_cols if df[c].nunique(dropna=True) <= 1]
        if const:
            df.drop(columns=const, inplace=True)

        return df

    # ------------------------------------------------------------------ results
    @property
    def results(self) -> Dict[str, Dict[str, Any]]:
        if self._results is None:
            raise RuntimeError("Call `.fit()` first.")
        return self._results