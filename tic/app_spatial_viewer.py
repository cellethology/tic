"""
Spatial Transcriptomics Viewer
==============================

Dash application for interactive exploration of spatial transcriptomics
(AnnData) data.  NEW in this version (2025‑06‑26):

* **Line‑plot click → cell‑id linkage**
  * Click any point on the **line plot** to automatically populate the
    *Search Cell ID* box and trigger neighbourhood zoom on the main
    scatter.
* **Line plot window** (added previously)
  * Select a **numeric** `obs` field as the x‑axis (e.g. `size`, `pseudotime`).
  * Select **one or more** genes / biomarkers as y‑axis variables.
  * `x_transform`: `'raw'`, `'bin'` (default, 100 bins), `'bin+normalize'`.
  * `y_transform`: `None` (no transform), `'normalize'`, `'smooth'`,
    `'normalize+smooth'`.
  * Built‑in helper utilities: `moving_average`, `normalize`,
    `fill_nan_with_interp`.

Other key features retained from earlier version:

* Cell‑type filtering, gene‑expression scatter, metadata field colouring.
* Cell search with k‑nearest neighbourhood mini‑map.
* Consistent palette & interactive tool‑tips.

Author: Zhang Jiahao  |  Project: TIC – Tumor Inference of Causality
License: MIT
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.spatial
from anndata import AnnData

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, dcc, html

###############################################################################
# Helper utilities – transforms & smoothing
###############################################################################

N_BINS: int = 100  # default number of bins for x‑axis discretisation


def moving_average(y: np.ndarray, window: int = 5) -> np.ndarray:  # noqa: D401
    """Return simple moving‑average (ignores NaNs)."""
    if window < 1 or window > len(y):
        raise ValueError("Window size must be between 1 and length of y.")
    y_filled = np.nan_to_num(y, nan=np.nanmean(y))
    kernel = np.ones(window) / window
    return np.convolve(y_filled, kernel, mode="same")


def normalize(y: np.ndarray) -> np.ndarray:  # noqa: D401
    """Min‑max scale to [0, 1] (ignores NaNs)."""
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    if ymax > ymin:
        return (y - ymin) / (ymax - ymin + 1e-12)
    return y


def fill_nan_with_interp(arr: np.ndarray) -> np.ndarray:  # noqa: D401
    """Fill NaNs in 1‑D array via linear interpolation."""
    x = np.arange(len(arr))
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return np.nan_to_num(arr, nan=0.0)
    return np.interp(x, x[mask], arr[mask])

###############################################################################
# Data loading / caching
###############################################################################


class DataStore:  # noqa: D101
    """Cached wrapper around AnnData with spatial coordinates & helpers."""

    def __init__(self, path: Path):
        self.adata: AnnData = sc.read(path)

        # spatial coordinates --------------------------------------------------
        if {"x", "y"}.issubset(self.adata.obs.columns):
            self.xy = self.adata.obs[["x", "y"]].to_numpy(dtype=np.float32)
        elif "spatial" in self.adata.obsm:
            self.xy = self.adata.obsm["spatial"][:, :2].astype(np.float32)
        else:
            raise KeyError(
                "Spatial coordinates ('x','y' columns or 'spatial' obsm) not found.")

        # k‑d tree for neighbourhood queries ----------------------------------
        self.kdtree = scipy.spatial.KDTree(self.xy)

        # cell‑type categorical & palette -------------------------------------
        self.cell_types = self.adata.obs["cell_type"].astype("category")
        palette = px.colors.qualitative.Plotly
        self.ct_colors = {
            ct: palette[i % len(palette)]
            for i, ct in enumerate(self.cell_types.cat.categories)
        }

        # obs keys and identify numeric ones ----------------------------------
        self.obs_keys: list[str] = list(self.adata.obs.columns)
        self.numeric_obs: list[str] = [
            k for k in self.obs_keys if pd.api.types.is_numeric_dtype(self.adata.obs[k])
        ]
        self.obs_minmax = {
            k: (self.adata.obs[k].min(), self.adata.obs[k].max()) for k in self.numeric_obs
        }


@lru_cache(maxsize=1)
def prepare(path: str | Path) -> DataStore:  # noqa: D401
    """Lazy‑load DataStore with LRU cache."""
    return DataStore(Path(path))

###############################################################################
# Dash application factory
###############################################################################


def make_app(h5ad_path: str | Path) -> Dash:  # noqa: WPS231
    ds = prepare(str(h5ad_path))
    adata, xy = ds.adata, ds.xy
    cell_types, ct_colors = ds.cell_types, ds.ct_colors

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    server = app.server  # type: ignore[attr-defined]

    # ---------- Dropdown options ---------------------------------------------
    ct_options = [{"label": ct, "value": ct} for ct in cell_types.cat.categories]
    obs_options = [{"label": k, "value": k} for k in ds.obs_keys]
    gene_options = [{"label": g, "value": g} for g in adata.var_names]
    numeric_obs_options = [{"label": k, "value": k} for k in ds.numeric_obs]

    # ---------- Layout -------------------------------------------------------
    app.layout = dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                [
                    # Sidebar -------------------------------------------------
                    dbc.Col(
                        [
                            html.H4("Spatial Transcriptomics Viewer"),
                            html.Hr(),

                            # FILTERS --------------------------------------
                            html.Label("Filter by Cell Type"),
                            dcc.Dropdown(
                                id="cell-type-dropdown",
                                options=ct_options,
                                multi=True,
                                value=[o["value"] for o in ct_options],
                            ),
                            html.Br(),

                            html.Label("Gene / Biomarker (scatter view)"),
                            dcc.Dropdown(
                                id="gene-dropdown",
                                options=gene_options,
                                placeholder="Enter gene symbol…",
                                clearable=True,
                            ),
                            html.Br(),

                            html.Label("Visualise `obs` Field (scatter view)"),
                            dcc.Dropdown(
                                id="obs-dropdown",
                                options=obs_options,
                                placeholder="Select metadata field…",
                                clearable=True,
                            ),
                            html.Br(),

                            html.Label("Normalise scalar to [0–1] (scatter view)"),
                            dcc.Checklist(
                                id="normalize-toggle",
                                options=[{"label": "Enable", "value": "norm"}],
                                value=[],
                                inputStyle={"margin-right": "6px"},
                            ),
                            html.Hr(),

                            # LINE PLOT CONFIG -----------------------------
                            html.H5("Line Plot Settings"),
                            html.Label("X-axis (numeric obs)"),
                            dcc.Dropdown(
                                id="line-x-dropdown",
                                options=numeric_obs_options,
                                value=(numeric_obs_options[0]["value"] if numeric_obs_options else None),
                            ),
                            html.Br(),

                            html.Label("Y-axis Genes / Biomarkers"),
                            dcc.Dropdown(
                                id="line-gene-dropdown",
                                options=gene_options,
                                multi=True,
                                placeholder="Select one or more genes…",
                            ),
                            html.Br(),

                            html.Label("x_transform"),
                            dcc.Dropdown(
                                id="x-transform-dropdown",
                                options=[
                                    {"label": "raw", "value": "raw"},
                                    {"label": "bin (100)", "value": "bin"},
                                    {"label": "bin+normalize", "value": "bin+normalize"},
                                ],
                                value="bin",
                                clearable=False,
                            ),
                            html.Br(),

                            html.Label("y_transform"),
                            dcc.Dropdown(
                                id="y-transform-dropdown",
                                options=[
                                    {"label": "none", "value": "none"},
                                    {"label": "normalize", "value": "normalize"},
                                    {"label": "smooth", "value": "smooth"},
                                    {"label": "normalize+smooth", "value": "normalize+smooth"},
                                ],
                                value="none",
                                clearable=False,
                            ),
                            html.Hr(),

                            # SEARCH --------------------------------------
                            html.Label("Search Cell ID"),
                            dcc.Input(
                                id="cell-search",
                                type="text",
                                placeholder="Enter cell obs_name…",
                                debounce=False,  # value change triggers immediately
                            ),
                            html.Small("  ← auto‑filled on line‑plot click"),
                            html.Br(),
                            html.Br(),

                            html.Label("Neighborhood Size (k)"),
                            dcc.Input(
                                id="neighborhood-k",
                                type="number",
                                min=1,
                                max=1000,
                                step=1,
                                value=30,
                            ),
                            html.Br(),
                            html.Br(),

                            html.Div(id="cell-info", className="mt-3"),
                        ],
                        width=3,
                        style={"overflowY": "scroll", "height": "95vh", "padding": "20px"},
                    ),

                    # MAIN PANEL -------------------------------------------
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="spatial-plot",
                                style={"height": "60vh"},
                                config={"displaylogo": False},
                            ),
                            html.Hr(),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H6("Neighborhood View (centre ±k)"),
                                            dcc.Graph(
                                                id="domain-plot",
                                                style={"height": "22vh"},
                                                config={"displaylogo": False},
                                            ),
                                        ],
                                        width=4,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H6("Line Plot"),
                                            dcc.Graph(
                                                id="line-plot",
                                                style={"height": "22vh"},
                                                config={"displaylogo": False},
                                            ),
                                        ],
                                        width=8,
                                    ),
                                ]
                            ),
                        ],
                        width=9,
                    ),
                ]
            )
        ],
    )

    ###############################################################################
    # Helper – create scatter trace
    ###############################################################################

    def _masked_scatter(
        xy_pts: np.ndarray,
        marker_kwargs: dict,
        hovertext: Iterable[str] | None = None,
        name: str | None = None,
        showlegend: bool = False,
    ) -> go.Scattergl:  # noqa: D401
        return go.Scattergl(
            x=xy_pts[:, 0],
            y=xy_pts[:, 1],
            mode="markers",
            marker=marker_kwargs,
            name=name,
            showlegend=showlegend,
            text=list(hovertext) if hovertext is not None else None,
            hovertemplate="Cell: %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
        )

    ###############################################################################
    # Callback – scatter + neighbourhood + domain mini‑map
    ###############################################################################

    @app.callback(
        Output("spatial-plot", "figure"),
        Output("cell-info", "children"),
        Output("domain-plot", "figure"),
        Input("cell-type-dropdown", "value"),
        Input("gene-dropdown", "value"),
        Input("obs-dropdown", "value"),
        Input("normalize-toggle", "value"),
        Input("cell-search", "value"),  # ← value change triggers update
        State("neighborhood-k", "value"),
    )
    def update_scatter(
        selected_cts: list[str],
        gene: str | None,
        obs_field: str | None,
        norm_flag: list[str],
        cell_id: str | None,
        k: int | None,
    ):  # noqa: C901, WPS231
        # --------------------------------------------------- helpers
        def maybe_normalise(arr: np.ndarray) -> np.ndarray:
            return normalize(arr) if "norm" in norm_flag else arr

        try:
            k = int(k) if k is not None else 30
        except (TypeError, ValueError):
            k = 30

        mask_all = cell_types.isin(selected_cts).to_numpy()
        fig = go.Figure()

        # --------------- colouring logic ------------------------------------
        if obs_field:  # metadata colouring
            col = adata.obs[obs_field]
            if pd.api.types.is_numeric_dtype(col):
                vals = col.to_numpy(dtype=float)
                vals_disp = maybe_normalise(vals)
                vmin, vmax = (
                    (0.0, 1.0) if "norm" in norm_flag else ds.obs_minmax[obs_field]
                )
                fig.add_trace(
                    _masked_scatter(
                        xy[mask_all],
                        {
                            "size": 5,
                            "opacity": 0.8,
                            "color": vals_disp[mask_all],
                            "colorscale": "Viridis",
                            "cmin": vmin,
                            "cmax": vmax,
                            "colorbar": {"title": f"{obs_field}{' (norm)' if 'norm' in norm_flag else ''}"},
                        },
                        hovertext=adata.obs_names[mask_all],
                    )
                )
            else:  # categorical
                cats = pd.Categorical(col).categories
                cat_colors = {
                    cat: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                    for i, cat in enumerate(cats)
                }
                for cat in cats:
                    m = mask_all & (col == cat).to_numpy()
                    if not m.any():
                        continue
                    fig.add_trace(
                        _masked_scatter(
                            xy[m],
                            {"size": 5, "opacity": 0.8, "color": cat_colors[cat]},
                            hovertext=adata.obs_names[m],
                            name=str(cat),
                            showlegend=True,
                        )
                    )
        elif gene:  # gene expression colouring
            idx = adata.var_names.get_loc(gene)
            expr = (
                adata.X[:, idx].A.ravel() if hasattr(adata.X, "A") else adata.X[:, idx]
            ).astype(float)
            expr_disp = maybe_normalise(expr)
            vmin, vmax = (
                (0.0, 1.0) if "norm" in norm_flag else (expr.min(), expr.max())
            )
            fig.add_trace(
                _masked_scatter(
                    xy[mask_all],
                    {
                        "size": 5,
                        "opacity": 0.8,
                        "color": expr_disp[mask_all],
                        "colorscale": "Viridis",
                        "cmin": vmin,
                        "cmax": vmax,
                        "colorbar": {"title": f"{gene}{' (norm)' if 'norm' in norm_flag else ''}"},
                    },
                    hovertext=adata.obs_names[mask_all],
                )
            )
        else:  # default cell‑type palette
            for ct in cell_types.cat.categories:
                m = mask_all & (cell_types == ct).to_numpy()
                if not m.any():
                    continue
                fig.add_trace(
                    _masked_scatter(
                        xy[m],
                        {"size": 5, "opacity": 0.8, "color": ct_colors[ct]},
                        hovertext=adata.obs_names[m],
                        name=str(ct),
                        showlegend=True,
                    )
                )

        # --------------- neighbourhood zoom ---------------------------------
        card = None
        domain_fig = go.Figure()
        if cell_id and cell_id in adata.obs_names:
            idx_centre = int(adata.obs_names.get_loc(cell_id))
            centre_xy = ds.xy[idx_centre]
            _, nn = ds.kdtree.query(centre_xy, k=k + 1)  # include self
            nn_xy = ds.xy[nn]
            nn_ct = cell_types.iloc[nn].to_numpy()

            # zoom main scatter
            fig.update_layout(
                xaxis=dict(range=[centre_xy[0] - 50, centre_xy[0] + 50]),
                yaxis=dict(range=[centre_xy[1] - 50, centre_xy[1] + 50]),
            )
            fig.add_trace(
                go.Scattergl(
                    x=[centre_xy[0]],
                    y=[centre_xy[1]],
                    mode="markers",
                    marker=dict(color="black", size=10, symbol="x"),
                    showlegend=False,
                )
            )

            # mini‑map domain ---------------------------------------------
            for ct in np.unique(nn_ct):
                mask_nn = nn_ct == ct
                domain_fig.add_trace(
                    go.Scattergl(
                        x=nn_xy[mask_nn, 0],
                        y=nn_xy[mask_nn, 1],
                        mode="markers",
                        marker=dict(size=6, color=ct_colors[ct], opacity=0.8),
                        name=ct,
                        showlegend=True,
                    )
                )
            domain_fig.add_trace(
                go.Scattergl(
                    x=[centre_xy[0]],
                    y=[centre_xy[1]],
                    mode="markers",
                    marker=dict(color="black", size=10, symbol="x"),
                    showlegend=False,
                )
            )
            domain_fig.update_layout(
                template="plotly_white",
                margin=dict(t=10, l=10, r=10, b=10),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=200,
            )
            domain_fig.update_yaxes(scaleanchor="x", scaleratio=1)

            # info card ----------------------------------------------------
            info = adata.obs.iloc[idx_centre]
            rows = [html.Tr([html.Td(str(k)), html.Td(str(v))]) for k, v in info.items()]
            card = dbc.Card(
                dbc.CardBody([
                    html.H6(f"Cell: {cell_id}"),
                    dbc.Table([
                        html.Thead(html.Tr([html.Th("Field"), html.Th("Value")])),
                        html.Tbody(rows),
                    ], bordered=True, size="sm", hover=True),
                    html.Small(f"Showing centre + {k} neighbours."),
                ]),
                className="mt-3",
            )

        # empty domain fig if not used ----------------------------------------
        if not domain_fig.data:
            domain_fig.update_layout(
                template="plotly_white",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(t=10, l=10, r=10, b=10),
                height=200,
            )

        # common scatter layout tweaks ----------------------------------------
        fig.update_layout(
            template="plotly_white",
            dragmode="pan",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(t=10, l=10, b=10, r=10),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig, card, domain_fig

    ###############################################################################
    # Callback – line plot
    ###############################################################################

    @app.callback(
        Output("line-plot", "figure"),
        Input("cell-type-dropdown", "value"),
        Input("line-x-dropdown", "value"),
        Input("line-gene-dropdown", "value"),
        Input("x-transform-dropdown", "value"),
        Input("y-transform-dropdown", "value"),
    )
    def update_line_plot(
        selected_cts: list[str],
        x_key: str | None,
        genes: list[str] | str | None,
        x_transform: Literal["raw", "bin", "bin+normalize"],
        y_transform: str,
    ):
        fig = go.Figure()
        if x_key is None or not genes:
            fig.update_layout(
                template="plotly_white",
                xaxis_title="x",
                yaxis_title="gene expression (a.u.)",
            )
            return fig

        if isinstance(genes, str):
            genes = [genes]

        mask_cts = cell_types.isin(selected_cts).to_numpy()
        X_raw = adata.obs[x_key].to_numpy(dtype=float)[mask_cts]

        # ------------------------------------------------- handle x‑transform
        if x_transform == "raw":
            sort_idx = np.argsort(X_raw)
            xx = X_raw[sort_idx]
        else:  # bin / bin+normalize
            bins = np.linspace(np.nanmin(X_raw), np.nanmax(X_raw), N_BINS + 1)
            bin_idx = np.digitize(X_raw, bins) - 1  # 0‑based
            bin_centres = (bins[:-1] + bins[1:]) / 2
            xx = bin_centres  # fixed len

        if x_transform.endswith("normalize"):
            xx = normalize(xx)
            x_label = f"{x_key} (norm)"
        else:
            x_label = x_key if x_transform == "raw" else f"{x_key} (binned)"

        # ------------------------------------------------- compute y for each gene
        for g in genes:
            idx = adata.var_names.get_loc(g)
            yvals_full = (
                adata.X[:, idx].A.ravel() if hasattr(adata.X, "A") else adata.X[:, idx]
            ).astype(float)[mask_cts]

            if x_transform == "raw":
                yvals = yvals_full[sort_idx]
            else:
                y_agg = np.full(N_BINS, np.nan, dtype=float)
                for b in range(N_BINS):
                    m = bin_idx == b
                    if m.any():
                        y_agg[b] = np.nanmean(yvals_full[m])
                yvals = fill_nan_with_interp(y_agg)

            # y‑transform --------------------------------------------------
            if y_transform in {"normalize", "normalize+smooth"}:
                yvals = normalize(yvals)
            if y_transform in {"smooth", "normalize+smooth"}:
                yvals = moving_average(yvals, window=max(3, len(yvals) // 50))

            fig.add_trace(
                go.Scatter(
                    x=xx,
                    y=yvals,
                    mode="lines+markers",
                    name=g,
                )
            )

        fig.update_layout(
            template="plotly_white",
            xaxis_title=x_label,
            yaxis_title="Expression (a.u.)",
            margin=dict(t=10, l=40, r=10, b=40),
            legend_title="Gene",
        )
        return fig

    ###############################################################################
    # Callback – line‑plot click → fill cell‑search
    ###############################################################################

    @app.callback(
        Output("cell-search", "value"),
        Input("line-plot", "clickData"),
        State("cell-type-dropdown", "value"),
        State("line-x-dropdown", "value"),
        State("line-gene-dropdown", "value"),
    )
    def update_cell_search(click_data, selected_cts, x_key, genes):
        if not click_data or "points" not in click_data:
            return dash.no_update

        point = click_data["points"][0]
        x_clicked = point["x"]
        y_clicked = point["y"]
        curve_idx = point["curveNumber"]

        if not genes or curve_idx >= len(genes):
            return dash.no_update

        gene_name = genes[curve_idx] if isinstance(genes, list) else genes

        mask_cts = cell_types.isin(selected_cts).to_numpy()
        x_vals = adata.obs[x_key].to_numpy(dtype=float)[mask_cts]
        y_vals = (
            adata.X[:, adata.var_names.get_loc(gene_name)].A.ravel()
            if hasattr(adata.X, "A")
            else adata.X[:, adata.var_names.get_loc(gene_name)]
        ).astype(float)[mask_cts]

        distances = np.abs(x_vals - x_clicked) + np.abs(y_vals - y_clicked)
        if np.isnan(distances).all():
            return dash.no_update

        min_idx = np.nanargmin(distances)
        obs_names_masked = adata.obs_names[mask_cts]
        return obs_names_masked[min_idx]

    return app

###############################################################################
# Entrypoint
###############################################################################

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python app_spatial_viewer.py /path/to/data.h5ad")
    make_app(sys.argv[1]).run(host="0.0.0.0", port=8050, debug=True)
