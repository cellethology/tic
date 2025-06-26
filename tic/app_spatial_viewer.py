# =============================================================
# Title: Spatial Transcriptomics Viewer (AnnData Dashboard)
# Description:
#     Interactive Dash application for visualizing spatial transcriptomics 
#     data stored in AnnData (.h5ad) format. Supports visualization of:
#       - Cell type annotations
#       - Gene/biomarker expression
#       - Metadata fields in `obs`
#       - Local neighborhood view of a queried cell
#
#     Main Features:
#       - Dropdown filtering by cell type or metadata field
#       - Gene expression heatmaps on spatial plots
#       - Cell ID search with zoom + k-nearest neighbors
#       - Mini-map window for local cell neighborhood with consistent coloring
#       - Optional 0–1 normalisation for any scalar (numeric) data
#       - Hover tool-tip shows Cell ID + spatial (x, y) coordinates
#
# Author: Zhang Jiahao
# Project: TIC - Tumor Inference of Causality in EMT Progression
# Affiliation: Westlake University, CELab
# Contact: jiahao.zhang.public@gmail.com
#
# Dependencies:
#     - Dash
#     - dash-bootstrap-components
#     - scanpy, anndata
#     - numpy, pandas, scipy, plotly
#
# Usage:
#     python tic/app_spatial_viewer.py /path/to/data.h5ad
#
# License: MIT
# =============================================================

from __future__ import annotations
import sys
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.spatial

from dash import Dash, html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px


# ---------------------------------------------------------------------
# Data loading & pre-processing
# ---------------------------------------------------------------------
class DataStore:
    def __init__(self, path: Path):
        adata = sc.read(path)

        # spatial coordinates
        if {"x", "y"}.issubset(adata.obs.columns):
            xy = adata.obs[["x", "y"]].to_numpy()
        elif "spatial" in adata.obsm:
            xy = adata.obsm["spatial"][:, :2]
        else:
            raise KeyError("Spatial coordinates not found in AnnData.")

        self.adata = adata
        self.xy = xy.astype(np.float32)
        self.kdtree = scipy.spatial.KDTree(self.xy)
        self.cell_types = adata.obs["cell_type"].astype("category")

        # default palette
        palette = px.colors.qualitative.Plotly
        self.ct_colors = {
            ct: palette[i % len(palette)]
            for i, ct in enumerate(self.cell_types.cat.categories)
        }

        # obs keys & numeric ranges
        self.obs_keys = list(adata.obs.columns)
        self.obs_minmax = {
            key: (adata.obs[key].min(), adata.obs[key].max())
            for key in self.obs_keys
            if pd.api.types.is_numeric_dtype(adata.obs[key])
        }


@lru_cache(maxsize=1)
def prepare(path: str | Path) -> DataStore:
    return DataStore(Path(path))


# ---------------------------------------------------------------------
# Dash layout & callbacks
# ---------------------------------------------------------------------
def make_app(h5ad_path: str | Path) -> Dash:
    ds = prepare(str(h5ad_path))
    adata, xy = ds.adata, ds.xy
    cell_types, ct_colors = ds.cell_types, ds.ct_colors

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    server = app.server  # noqa: F841

    # dropdown options
    ct_options = [{"label": ct, "value": ct} for ct in cell_types.cat.categories]
    obs_options = [{"label": key, "value": key} for key in ds.obs_keys]
    gene_options = [{"label": g, "value": g} for g in adata.var_names]

    # -----------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------
    app.layout = dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                [
                    # Sidebar
                    dbc.Col(
                        [
                            html.H4("Spatial Transcriptomics Viewer"),
                            html.Hr(),

                            html.Label("Filter by Cell Type"),
                            dcc.Dropdown(
                                id="cell-type-dropdown",
                                options=ct_options,
                                multi=True,
                                value=[o["value"] for o in ct_options],
                            ),
                            html.Br(),

                            html.Label("Gene / Biomarker"),
                            dcc.Dropdown(
                                id="gene-dropdown",
                                options=gene_options,
                                placeholder="Enter gene symbol…",
                                clearable=True,
                            ),
                            html.Br(),

                            html.Label("Visualise `obs` Field"),
                            dcc.Dropdown(
                                id="obs-dropdown",
                                options=obs_options,
                                placeholder="Select metadata field…",
                                clearable=True,
                            ),
                            html.Br(),

                            html.Label("Normalise scalar to [0–1]"),
                            dcc.Checklist(
                                id="normalize-toggle",
                                options=[{"label": "Enable", "value": "norm"}],
                                value=[],
                                inputStyle={"margin-right": "6px"},
                            ),
                            html.Br(),

                            html.Label("Search Cell ID"),
                            dcc.Input(
                                id="cell-search",
                                type="text",
                                placeholder="Enter cell obs_name…",
                                debounce=True,
                            ),
                            html.Small("Press Enter to zoom."),
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
                    ),

                    # Main panel
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="spatial-plot",
                                style={"height": "70vh"},
                                config={"displaylogo": False},
                            ),
                            html.Hr(),
                            html.H6("Neighborhood View (centre ±k)"),
                            dcc.Graph(
                                id="domain-plot",
                                style={"height": "18vh"},
                                config={"displaylogo": False},
                            ),
                        ],
                        width=9,
                    ),
                ]
            )
        ],
    )

    # -----------------------------------------------------------------
    # Helper for scatter traces
    # -----------------------------------------------------------------
    def masked_scatter(xy_pts, mask_bool, marker_kwargs, name=None, showlegend=False):
        # xy_pts should be xy[mask_bool]
        return go.Scattergl(
            x=xy_pts[:, 0],
            y=xy_pts[:, 1],
            mode="markers",
            marker=marker_kwargs,
            name=name,
            showlegend=showlegend,
            text=adata.obs_names[mask_bool],
            hovertemplate="Cell: %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
        )

    # -----------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------
    @app.callback(
        Output("spatial-plot", "figure"),
        Output("cell-info", "children"),
        Output("domain-plot", "figure"),
        Input("cell-type-dropdown", "value"),
        Input("gene-dropdown", "value"),
        Input("obs-dropdown", "value"),
        Input("normalize-toggle", "value"),
        Input("cell-search", "n_submit"),
        State("cell-search", "value"),
        State("neighborhood-k", "value"),
    )
    def update_plots(selected_cts, gene, obs_field, norm_flag, n_submit, cell_id, k):
        def maybe_normalise(arr: np.ndarray) -> np.ndarray:
            if "norm" in norm_flag:
                rng = arr.max() - arr.min()
                if rng > 0:
                    return (arr - arr.min()) / rng
            return arr

        # validate k
        try:
            k = int(k) if k is not None else 30
        except (ValueError, TypeError):
            k = 30

        mask_all = cell_types.isin(selected_cts).to_numpy()
        fig = go.Figure()

        # Colour logic
        if obs_field:
            col = adata.obs[obs_field]
            if pd.api.types.is_numeric_dtype(col):
                vals = col.to_numpy()
                vals_disp = maybe_normalise(vals)
                vmin = 0 if "norm" in norm_flag else ds.obs_minmax[obs_field][0]
                vmax = 1 if "norm" in norm_flag else ds.obs_minmax[obs_field][1]
                fig.add_trace(
                    masked_scatter(
                        xy[mask_all],
                        mask_all,
                        {
                            "size": 5,
                            "opacity": 0.8,
                            "color": vals_disp[mask_all],
                            "colorscale": "Viridis",
                            "cmin": vmin,
                            "cmax": vmax,
                            "colorbar": {
                                "title": f"{obs_field}{' (norm)' if 'norm' in norm_flag else ''}"
                            },
                        },
                    )
                )
            else:
                cats = pd.Categorical(col).categories
                obs_colors = {
                    cat: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                    for i, cat in enumerate(cats)
                }
                for cat in cats:
                    m = mask_all & (col == cat).to_numpy()
                    if not m.any():
                        continue
                    fig.add_trace(
                        masked_scatter(
                            xy[m],
                            m,
                            {"size": 5, "opacity": 0.8, "color": obs_colors[cat]},
                            name=str(cat),
                            showlegend=True,
                        )
                    )

        elif gene:
            idx = adata.var_names.get_loc(gene)
            expr = (
                adata.X[:, idx].A.ravel()
                if hasattr(adata.X, "A")
                else adata.X[:, idx]
            )
            expr = expr.astype(np.float64)
            expr_disp = maybe_normalise(expr)
            vmin = 0 if "norm" in norm_flag else expr.min()
            vmax = 1 if "norm" in norm_flag else expr.max()
            fig.add_trace(
                masked_scatter(
                    xy[mask_all],
                    mask_all,
                    {
                        "size": 5,
                        "opacity": 0.8,
                        "color": expr_disp[mask_all],
                        "colorscale": "Viridis",
                        "cmin": vmin,
                        "cmax": vmax,
                        "colorbar": {
                            "title": f"{gene}{' (norm)' if 'norm' in norm_flag else ''}"
                        },
                    },
                )
            )

        else:
            for ct in cell_types.cat.categories:
                m = mask_all & (cell_types == ct).to_numpy()
                if not m.any():
                    continue
                fig.add_trace(
                    masked_scatter(
                        xy[m],
                        m,
                        {"size": 5, "opacity": 0.8, "color": ct_colors[ct]},
                        name=str(ct),
                        showlegend=True,
                    )
                )

        # Neighbourhood & Info card
        card = None
        domain_fig = go.Figure()
        if cell_id and cell_id in adata.obs_names:
            idx = adata.obs_names.get_loc(cell_id)
            centre = ds.xy[idx]
            _, nn = ds.kdtree.query(centre, k=k + 1)
            nn_xy = ds.xy[nn]

            # zoom main
            fig.update_layout(
                xaxis=dict(range=[centre[0] - 50, centre[0] + 50]),
                yaxis=dict(range=[centre[1] - 50, centre[1] + 50]),
            )
            fig.add_trace(
                go.Scattergl(
                    x=[centre[0]],
                    y=[centre[1]],
                    mode="markers",
                    marker=dict(color="black", size=10, symbol="x"),
                    showlegend=False,
                )
            )

            # domain
            nn_ct = cell_types.iloc[nn].to_numpy()
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
                    x=[centre[0]],
                    y=[centre[1]],
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

            # info card
            info = adata.obs.iloc[idx]
            rows = [html.Tr([html.Td(str(k)), html.Td(str(v))]) for k, v in info.items()]
            card = dbc.Card(
                dbc.CardBody(
                    [
                        html.H6(f"Cell: {cell_id}"),
                        dbc.Table(
                            [html.Thead(html.Tr([html.Th("Field"), html.Th("Value")])), html.Tbody(rows)],
                            bordered=True, size="sm", hover=True,
                        ),
                        html.Small(f"Showing centre + {k} neighbours."),
                    ]
                ),
                className="mt-3",
            )

        if not domain_fig.data:
            domain_fig.update_layout(
                template="plotly_white",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(t=10, l=10, r=10, b=10),
                height=200,
            )

        fig.update_layout(
            template="plotly_white",
            dragmode="pan",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(t=10, l=10, b=10, r=10),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig, card, domain_fig

    return app


# Entrypoint
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python tic/app_spatial_viewer.py /path/to/data.h5ad")
    make_app(sys.argv[1]).run(host="0.0.0.0", port=8050, debug=True)