# tic/app_spatial_viewer.py
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
#
# Author: Zhang Jiahao
# Project: TIC - Tumor Inference of Causality in EMT Progression
# Affiliation: Westlake University, CELab
# Contact: jiahao.zhang.public@gmail.com

# Dependencies:
#     - Dash
#     - dash-bootstrap-components
#     - scanpy, anndata
#     - numpy, pandas, scipy, plotly
#
# Usage:
#     python tic/app_spatial_viewer.py /path/to/data.h5ad
#
# License: MIT (or specify your license)
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
    server = app.server  # noqa: F841  (needed for gunicorn / Heroku)

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
                    # ----------------------  sidebar  ---------------------
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

                    # ----------------------  main panel  ------------------
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
    def masked_scatter(xy_, mask_, marker_kwargs, name=None, showlegend=False):
        return go.Scattergl(
            x=xy_[mask_, 0],
            y=xy_[mask_, 1],
            mode="markers",
            marker=marker_kwargs,
            name=name,
            showlegend=showlegend,
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
        Input("cell-search", "n_submit"),
        State("cell-search", "value"),
        State("neighborhood-k", "value"),
    )
    def update_plots(selected_cts, gene, obs_field, n_submit, cell_id, k):
        # k may be None / non-int
        try:
            k = int(k) if k is not None else 30
        except (ValueError, TypeError):
            k = 30

        mask = cell_types.isin(selected_cts)
        fig = go.Figure()

        # --------------------  colouring logic  -----------------------
        if obs_field:  # visualise selected obs column
            col = adata.obs[obs_field]
            if pd.api.types.is_numeric_dtype(col):
                vals = col.to_numpy()
                vmin, vmax = ds.obs_minmax[obs_field]
                fig.add_trace(
                    masked_scatter(
                        xy,
                        mask.to_numpy(),
                        {
                            "size": 5,
                            "opacity": 0.8,
                            "color": vals[mask],
                            "colorscale": "Viridis",
                            "cmin": vmin,
                            "cmax": vmax,
                            "colorbar": {"title": obs_field},
                        },
                    )
                )
            else:  # categorical obs
                cats = pd.Categorical(col).categories
                palette = px.colors.qualitative.Plotly
                obs_colors = {
                    cat: palette[i % len(palette)] for i, cat in enumerate(cats)
                }
                for cat in cats:
                    m = mask & (col == cat)
                    if not m.any():
                        continue
                    fig.add_trace(
                        masked_scatter(
                            xy,
                            m.to_numpy(),
                            {"size": 5, "opacity": 0.8, "color": obs_colors[cat]},
                            name=str(cat),
                            showlegend=True,
                        )
                    )

        elif gene:  # visualise gene expression
            idx = adata.var_names.get_loc(gene)
            expr = (
                adata.X[:, idx].A.ravel()
                if hasattr(adata.X, "A")
                else adata.X[:, idx]
            )
            fig.add_trace(
                masked_scatter(
                    xy,
                    mask.to_numpy(),
                    {
                        "size": 5,
                        "opacity": 0.8,
                        "color": expr[mask],
                        "colorscale": "Viridis",
                        "cmin": expr.min(),
                        "cmax": expr.max(),
                        "colorbar": {"title": gene},
                    },
                )
            )

        else:  # default: colour by cell type
            for ct in cell_types.cat.categories:
                m = mask & (cell_types == ct)
                if not m.any():
                    continue
                fig.add_trace(
                    masked_scatter(
                        xy,
                        m.to_numpy(),
                        {"size": 5, "opacity": 0.8, "color": ct_colors[ct]},
                        name=str(ct),
                        showlegend=True,
                    )
                )

        # --------------------  neighbourhood logic  -------------------
        card = None
        domain_fig = go.Figure()

        if cell_id and cell_id in adata.obs_names:
            idx = adata.obs_names.get_loc(cell_id)
            centre = ds.xy[idx]

            # query k+1 because the first neighbour is the point itself
            _, nn = ds.kdtree.query(centre, k=k + 1)
            nn_xy = ds.xy[nn]

            # -- main plot: zoom to neighbourhood & mark centre
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
                    name="Center Cell",
                    showlegend=False,
                )
            )

            # -- neighbourhood (domain) plot: color by cell type consistent with main plot
            nn_cell_types = cell_types.iloc[nn].to_numpy()
            for ct in np.unique(nn_cell_types):
                mask = nn_cell_types == ct
                domain_fig.add_trace(
                    go.Scattergl(
                        x=nn_xy[mask, 0],
                        y=nn_xy[mask, 1],
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
                    marker=dict(size=10, color="black", symbol="x"),
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

            # -- info card
            cell_info = adata.obs.iloc[idx]
            rows = [
                html.Tr([html.Td(str(k)), html.Td(str(v))])
                for k, v in cell_info.items()
            ]
            card = dbc.Card(
                dbc.CardBody(
                    [
                        html.H6(f"Cell: {cell_id}"),
                        dbc.Table(
                            [
                                html.Thead(
                                    html.Tr([html.Th("Field"), html.Th("Value")])
                                ),
                                html.Tbody(rows),
                            ],
                            bordered=True,
                            size="sm",
                            hover=True,
                        ),
                        html.Small(f"Showing centre + {k} neighbours."),
                    ]
                ),
                className="mt-3",
            )

        # when no cell selected → empty neighbourhood figure
        if domain_fig.data == ():
            domain_fig.update_layout(
                template="plotly_white",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(t=10, l=10, r=10, b=10),
                height=200,
            )

        # consistent look
        fig.update_layout(
            template="plotly_white",
            dragmode="pan",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(t=10, l=10, b=10, r=10),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig, card, domain_fig

    # -----------------------------------------------------------------
    return app


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python app.py path/to/data.h5ad")
    make_app(sys.argv[1]).run(host="0.0.0.0", port=8050, debug=True)