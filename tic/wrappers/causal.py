# file: tic/wrappers/causal.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from ..constant import DEFAULT_KEY
from ..data.utils import get_cell_types, get_biomarkers
from ..causal.factory import CausalMethodFactory
from ..plotting import plot_causal_heatmap, plot_causal_bar, plot_causal_volcano


class CausalWrapper:
    """One-stop causal inference + plotting interface, now supports generic obsm features."""
    def __init__(
        self,
        *,
        outcome: str,
        feature_key: str,
        feature_names: Optional[Sequence[str]] = None,
        method: str = "granger_causality",
        bins: int | None = 100,
        method_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        outcome
            outcome biomarker name in adata.var_names
        feature_key
            key in adata.obsm for the predictor matrix (shape = n_obs × n_features)
        feature_names
            names for each column in that matrix;
            if None, we assume a celltype×gene matrix and auto-generate via get_cell_types/get_biomarkers
        method
            causal method
        bins
            number of pseudotime bins (None or <=1 disables binning)
        """
        self.outcome = outcome
        self.feature_key = feature_key
        self.feature_names = feature_names
        self.method = method
        self.bins = bins
        self.method_kwargs = method_kwargs or {}
        self._results: Dict[str, Dict[str, Any]] | None = None

    def fit(self, adata: AnnData) -> Dict[str, Dict[str, Any]]:
        """Run causal inference and store in .uns."""
        df = self._prepare_dataframe(adata)
        predictors = [c for c in df.columns if c not in ("time", "Y")]

        results: Dict[str, Dict[str, Any]] = {}
        for pred in predictors:
            from ..causal.causal_input import CausalInput

            ci = CausalInput(
                data=df[["Y", pred]].dropna(),
                treatment_col=pred,
                outcome_col="Y",
            )
            method_obj = CausalMethodFactory.get_method(self.method, **self.method_kwargs)
            method_obj.fit(ci)
            results[pred] = method_obj.estimate_effect(ci)

        self._results = results
        adata.uns[DEFAULT_KEY.get("causal_results")] = results
        return results

    def _prepare_dataframe(self, adata: AnnData) -> pd.DataFrame:
        """
        Aggregate into a DataFrame with columns:
          - time: bin centre
          - Y: mean outcome expression
          - <feature_names[i]>: mean of obsm[:, i] per bin
        """
        # 1. sanity checks
        if DEFAULT_KEY.get('pseudotime') not in adata.obs:
            raise KeyError("Run pseudotime first: missing .obs['pseudotime']")
        if self.outcome not in adata.var_names:
            raise ValueError(f"Outcome '{self.outcome}' not in .var_names")
        if self.feature_key not in adata.obsm:
            raise KeyError(f".obsm['{self.feature_key}'] not found")

        # 2. load data
        pt = adata.obs[DEFAULT_KEY.get('pseudotime')].to_numpy()
        mat = np.asarray(adata.obsm[self.feature_key])
        if mat.ndim == 1:
            mat = mat[:, None]
        n_feat = mat.shape[1]

        # 3. determine column names
        if self.feature_names is not None:
            if len(self.feature_names) != n_feat:
                raise ValueError(
                    f"feature_names length ({len(self.feature_names)}) "
                    f"!= number of columns in obsm ({n_feat})"
                )
            feat_names = list(self.feature_names)
        else:
            # assume celltype×gene layout
            cell_types = get_cell_types(adata)
            genes = get_biomarkers(adata)

            def _make_feature_names(cell_types: Sequence[str], genes: Sequence[str]) -> List[str]:
                """Return ``count_<cell>_<gene>`` for all combinations (row‑major)."""
                return [f"count_{ct}_{g}" for ct in cell_types for g in genes]
            
            feat_names = _make_feature_names(cell_types, genes)

            if len(feat_names) != n_feat:
                raise ValueError(
                    "obsm matrix columns != len(cell_types)*len(genes). "
                    "Either provide feature_names or use the matching obsm key."
                )

        # 4. find outcome index
        var_list = list(adata.var_names)
        y_idx = var_list.index(self.outcome)

        # 5. bin pseudotime
        if self.bins is None or self.bins <= 1:
            bin_ids = np.zeros_like(pt, dtype=int)
            centers = np.array([pt.mean()])
        else:
            edges = np.linspace(pt.min(), pt.max(), self.bins + 1)
            bin_ids = np.digitize(pt, edges) - 1
            bin_ids[bin_ids == self.bins] = self.bins - 1
            centers = (edges[:-1] + edges[1:]) / 2

        # 6. aggregate
        rows = []
        for b in np.unique(bin_ids):
            idx = np.where(bin_ids == b)[0]
            if idx.size == 0:
                continue
            y_mean = float(adata.X[idx, y_idx].mean())
            feat_mean = mat[idx].mean(axis=0)
            row = {"time": float(centers[b]), "Y": y_mean}
            row.update({name: float(val) for name, val in zip(feat_names, feat_mean)})
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

        # 7. drop constant predictors
        pred_cols = [c for c in df.columns if c not in ("time", "Y")]
        const_cols = [c for c in pred_cols if df[c].nunique(dropna=True) <= 1]
        if const_cols:
            df.drop(columns=const_cols, inplace=True)

        return df

    def plot(self, adata: AnnData, kind: str = "heatmap", **kwargs) -> None:
        fn_map = {
            "heatmap": plot_causal_heatmap,
            "bar": plot_causal_bar,
            "volcano": plot_causal_volcano,
        }
        if kind not in fn_map:
            raise ValueError(f"Unknown plot kind '{kind}'")
        fn_map[kind](adata, **kwargs)

    @property
    def results(self) -> Dict[str, Dict[str, Any]]:
        if self._results is None:
            raise RuntimeError("Run `.fit()` first.")
        return self._results