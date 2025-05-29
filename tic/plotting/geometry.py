# tic/plotting/geometry.py
"""
Geometry plotting utilities for the TIC pipeline.

This module provides functions to:
  - Extract cell boundary polygons from AnnData.
  - Normalize and center boundaries for analysis.
  - Perform Fourier-based shape reconstruction and descriptor computation.
  - Visualize original and reconstructed shapes.
  - Generate animated GIFs showing reconstruction error vs. number of frequencies.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from anndata import AnnData
from pathlib import Path
from typing import Sequence, Union, Callable, Literal

from ..features.geometry.fourier import (
    normalize_boundary,
    reconstruct_from_fourier,
    _METRICS as METRICS,
)


def get_cell_boundary_points(
    adata: AnnData,
    cell_id: str,
    boundary_type: Literal["cell", "nucleus"] = "cell",
) -> np.ndarray:
    """
    Extract the boundary vertices for a given cell and ensure the polygon is closed.

    Parameters
    ----------
    adata : AnnData
        Annotated data containing cell boundaries in adata.uns['cell_boundaries'].
    cell_id : str
        Identifier of the cell whose boundary is to be extracted.
    boundary_type : {'cell', 'nucleus'}, default 'cell'
        Type of boundary to retrieve: the full cell or just the nucleus.

    Returns
    -------
    pts : numpy.ndarray, shape (N, 2)
        Array of (x, y) coordinates for the polygon vertices, with the first point
        repeated as the last to close the polygon.

    Raises
    ------
    ValueError
        If fewer than 3 vertices are found for the specified cell.
    """
    df = adata.uns["cell_boundaries"][boundary_type]
    pts = df[df["cell_id"] == cell_id][["vertex_x", "vertex_y"]].values
    if pts.shape[0] < 3:
        raise ValueError(f"Insufficient boundary points for cell '{cell_id}'")
    # Close polygon if not already closed
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def plot_cell_boundary(
    adata: AnnData,
    cell_index: Union[int, str],
    ax: plt.Axes | None = None,
    boundary_type: Literal["cell", "nucleus"] = "cell",
    show: bool = True,
    **plot_kwargs,
) -> plt.Axes:
    """
    Plot the raw boundary polygon of a specified cell.

    Parameters
    ----------
    adata : AnnData
        Annotated data with cell boundaries.
    cell_index : int or str
        If int, index into adata.obs_names; if str, interprets directly as cell_id.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw. If None, a new figure and axes are created.
    boundary_type : {'cell', 'nucleus'}, default 'cell'
        Which boundary type to plot.
    show : bool, default True
        Whether to call plt.show() after plotting.
    plot_kwargs : dict
        Additional keyword arguments passed to ax.plot().

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the plot.
    """
    # Resolve cell_id
    cell_id = (
        adata.obs_names[cell_index]
        if isinstance(cell_index, int)
        else str(cell_index)
    )
    pts = get_cell_boundary_points(adata, cell_id, boundary_type)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(pts[:, 0], pts[:, 1], **plot_kwargs)
    ax.set_title(f"Cell ID: {cell_id}")
    ax.set_aspect("equal")
    ax.axis("off")

    if show:
        plt.show()
    return ax


def plot_fourier_normalized_shape(
    adata: AnnData,
    cell_index: Union[int, str],
    n_freq: int = 10,
    boundary_type: Literal["cell", "nucleus"] = "cell",
    normalize_area: bool = True,
) -> None:
    """
    Display the normalized cell shape, its Fourier magnitude spectrum, and reconstruction.

    Parameters
    ----------
    adata : AnnData
        Annotated data with cell boundaries.
    cell_index : int or str
        Cell index or cell_id as in plot_cell_boundary.
    n_freq : int, default 10
        Number of low-frequency components to keep for reconstruction and spectrum.
    boundary_type : {'cell', 'nucleus'}, default 'cell'
        Boundary type to analyze.
    normalize_area : bool, default True
        Whether to normalize polygon area to unity before analysis.

    Returns
    -------
    None
        Displays a 1x3 plot: normalized shape, spectrum, and reconstruction.
    """
    # Resolve and normalize
    cell_id = (
        adata.obs_names[cell_index]
        if isinstance(cell_index, int)
        else str(cell_index)
    )
    pts = get_cell_boundary_points(adata, cell_id, boundary_type)
    pts_norm = normalize_boundary(pts, normalize_area)
    z = pts_norm[:, 0] + 1j * pts_norm[:, 1]
    recon_pts = reconstruct_from_fourier(z, n_freq)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # A: Normalized Shape
    axs[0].plot(pts_norm[:, 0], pts_norm[:, 1])
    axs[0].set_title("A. Normalized Shape")
    axs[0].set_aspect("equal")
    axs[0].axis("off")

    # B: Fourier Spectrum
    Z = np.fft.fft(z)
    freqs = np.arange(1, len(Z) // 2)
    mags = np.abs(Z[1 : len(freqs) + 1])
    axs[1].stem(freqs[:n_freq], mags[:n_freq], basefmt=" ")
    axs[1].set_title("B. Fourier Magnitude Spectrum")
    axs[1].set_xlabel("Frequency k")
    axs[1].set_ylabel("|Z_k|")
    axs[1].grid(True)

    # C: Reconstruction
    axs[2].plot(recon_pts[:, 0], recon_pts[:, 1])
    axs[2].set_title(f"C. Reconstruction (k={n_freq})")
    axs[2].set_aspect("equal")
    axs[2].axis("off")

    fig.suptitle(f"Cell ID: {cell_id}", fontsize=14)
    plt.tight_layout()
    plt.show()


def generate_fourier_reconstruction_gif(
    adata: AnnData,
    cell_index: Union[int, str],
    ks: Sequence[int] | None = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "rmse",
    boundary_type: Literal["cell", "nucleus"] = "cell",
    normalize_area: bool = True,
    fps: int = 2,
    dpi: int = 100,
    save_path: Union[str, Path] = "fourier_reconstruction.gif",
) -> Path:
    """
    Create and save an animated GIF showing reconstruction error vs. number of frequencies.

    Parameters
    ----------
    adata : AnnData
        Annotated data with cell boundaries.
    cell_index : int or str
        Cell index or cell_id for analysis.
    ks : sequence of int, optional
        List of k values (number of low frequencies) to iterate over.
        Defaults to 1..(N/2-1).
    metric : {'rmse', 'mae'} or callable, default 'rmse'
        Error metric for reconstruction quality.
    boundary_type : {'cell', 'nucleus'}, default 'cell'
        Boundary type to analyze.
    normalize_area : bool, default True
        Whether to normalize polygon area before analysis.
    fps : int, default 2
        Frames per second for the output GIF.
    dpi : int, default 100
        Resolution of the saved GIF.
    save_path : str or Path, default 'fourier_reconstruction.gif'
        Path where the GIF will be saved (ensured to end with .gif).

    Returns
    -------
    out_path : Path
        Path to the saved GIF file.
    """
    # Resolve cell_id and prepare data
    cell_id = (
        adata.obs_names[cell_index]
        if isinstance(cell_index, int)
        else str(cell_index)
    )
    pts_orig = normalize_boundary(
        get_cell_boundary_points(adata, cell_id, boundary_type), normalize_area
    )
    z_orig = pts_orig[:, 0] + 1j * pts_orig[:, 1]
    n = len(z_orig)

    # Determine ks
    if ks is None:
        ks = list(range(1, n // 2))
    ks = sorted({int(k) for k in ks if 0 < k < n // 2})

    # Metric function lookup
    if isinstance(metric, str):
        metric_fn = METRICS.get(metric.lower())
        if metric_fn is None:
            raise ValueError(f"Unsupported metric '{metric}'")
        metric_name = metric.upper()
    else:
        metric_fn = metric
        metric_name = getattr(metric, "__name__", "metric")

    # Set up figure
    fig, (ax_shape, ax_curve) = plt.subplots(
        1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 2]}
    )
    # Plot original shape in gray
    ax_shape.plot(pts_orig[:, 0], pts_orig[:, 1], lw=1, color="lightgray")
    recon_line, = ax_shape.plot([], [], lw=2)
    ax_shape.set_aspect("equal"); ax_shape.axis("off")

    ax_curve.set_xlabel("k")
    ax_curve.set_ylabel(metric_name)
    ax_curve.grid(True)
    err_line, = ax_curve.plot([], [], marker="o")
    current_pt = ax_curve.scatter([], [], color="red", zorder=5)

    errors: list[float] = []

    def init():
        recon_line.set_data([], [])
        err_line.set_data([], [])
        current_pt.set_offsets(np.empty((0, 2)))
        ax_curve.set_xlim(0, max(ks))
        ax_curve.set_ylim(0, 1)
        return recon_line, err_line, current_pt

    def update(i: int):
        k = ks[i]
        pts_rec = reconstruct_from_fourier(z_orig, k)
        recon_line.set_data(pts_rec[:, 0], pts_rec[:, 1])
        err = metric_fn(pts_orig, pts_rec)
        errors.append(err)
        err_line.set_data(ks[: i + 1], errors)
        current_pt.set_offsets([[k, err]])
        ax_shape.set_title(f"Cell {cell_id} — k={k} ({metric_name}={err:.4f})")
        if i == 0:
            ax_curve.set_ylim(0, max(errors) * 1.05)
        return recon_line, err_line, current_pt

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(ks),
        init_func=init,
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )
    out_path = Path(save_path).with_suffix(".gif")
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"GIF saved ➜ {out_path.resolve()}")
    return out_path
