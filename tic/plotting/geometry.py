# tic/plotting/geometry.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from anndata import AnnData
from pathlib import Path
from typing import Sequence, Callable, Union, Literal

# -------------------------------------------------------------------
# Helper: Boundary extraction and normalization
# -------------------------------------------------------------------
def get_cell_boundary_points(
    adata: AnnData,
    cell_id: str,
    boundary_type: str = "cell"
) -> np.ndarray:
    """Extract (x, y) points for a given cell_id and close polygon."""
    df = adata.uns["cell_boundaries"][boundary_type]
    pts = df[df["cell_id"] == cell_id][["vertex_x", "vertex_y"]].values
    if pts.shape[0] < 3:
        raise ValueError(f"Insufficient points for {cell_id}")
    # Close polygon
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def normalize_boundary(
    pts: np.ndarray,
    normalize_area: bool = True
) -> np.ndarray:
    """Center boundary at origin and optionally normalize polygon area."""
    pts = pts - pts.mean(axis=0)
    if normalize_area:
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area > 0:
            pts = pts / np.sqrt(area)
    return pts

# -------------------------------------------------------------------
# Helper: Fourier reconstruction
# -------------------------------------------------------------------
def reconstruct_from_fourier(
    z: np.ndarray,
    k: int
) -> np.ndarray:
    """Keep k low frequencies and perform inverse FFT to reconstruct shape."""
    Z = np.fft.fft(z)
    if k < len(Z) // 2:
        Z[k+1 : -k] = 0
    recon = np.fft.ifft(Z)
    return np.vstack([recon.real, recon.imag]).T

# -------------------------------------------------------------------
# Metrics for error evaluation
# -------------------------------------------------------------------
def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))

def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return np.mean(np.sum(np.abs(a - b), axis=1))

_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmse": _rmse,
    "mae": _mae,
}

# -------------------------------------------------------------------
# Plotting: Boundaries and Fourier analysis
# -------------------------------------------------------------------

def plot_cell_boundary(
    adata: AnnData,
    cell_index: Union[int, str],
    ax: plt.Axes = None,
    boundary_type: str = "cell",
    show: bool = True,
    **plot_kwargs
) -> plt.Axes:
    """
    Plot the polygon for a given cell boundary.
    Parameters
    ----------
    adata: AnnData
    cell_index: int or str, the index of the cell to plot
    ax: plt.Axes, the axes to plot on
    boundary_type: str, the type of boundary to plot
    show: bool, whether to show the plot
    plot_kwargs: dict

    Returns
    -------
    ax: plt.Axes
    """
    # Resolve cell_id
    cell_id = adata.obs_names[cell_index] if isinstance(cell_index, int) else str(cell_index)
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
    boundary_type: str = "cell",
    normalize_area: bool = True
):
    """
    Show original normalized shape, its Fourier magnitude spectrum, and reconstruction.

    Parameters
    ----------
    adata: AnnData
    cell_index: int or str, the index of the cell to plot
    n_freq: int, the number of frequencies to plot
    boundary_type: str, the type of boundary to plot
    normalize_area: bool, whether to normalize the area of the boundary

    Returns
    -------
    None
    """
    # Resolve cell_id and get points
    cell_id = adata.obs_names[cell_index] if isinstance(cell_index, int) else str(cell_index)
    pts = get_cell_boundary_points(adata, cell_id, boundary_type)
    pts_norm = normalize_boundary(pts, normalize_area)
    z = pts_norm[:, 0] + 1j * pts_norm[:, 1]
    recon_pts = reconstruct_from_fourier(z, n_freq)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # A: Normalized shape
    axs[0].plot(pts_norm[:, 0], pts_norm[:, 1])
    axs[0].set_title("A. Normalized Shape")
    axs[0].set_aspect("equal")
    axs[0].axis("off")

    # B: Fourier spectrum
    Z = np.fft.fft(z)
    freqs = np.arange(1, len(Z)//2)
    mags = np.abs(Z[1:len(freqs)+1])
    axs[1].stem(freqs[:n_freq], mags[:n_freq], basefmt=" ")
    axs[1].set_title("B. Fourier Magnitude Spectrum")
    axs[1].set_xlabel("Frequency Index k")
    axs[1].set_ylabel("|Z_k|")
    axs[1].grid(True)

    # C: Reconstruction
    axs[2].plot(recon_pts[:, 0], recon_pts[:, 1])
    axs[2].set_title(f"C. Reconstructed (k={n_freq})")
    axs[2].set_aspect("equal")
    axs[2].axis("off")

    fig.suptitle(f"Cell ID: {cell_id}", fontsize=14)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# Animation: Fourier reconstruction error GIF
# -------------------------------------------------------------------

def generate_fourier_reconstruction_gif(
    adata: AnnData,
    cell_index: Union[int, str],
    ks: Sequence[int] | None = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "rmse",
    boundary_type: str = "cell",
    normalize_area: bool = True,
    fps: int = 2,
    dpi: int = 100,
    save_path: Union[str, Path] = "fourier_reconstruction.gif",
) -> Path:
    """
    Create and save a GIF showing reconstruction error vs. k frequency.

    Parameters
    ----------
    adata: AnnData
    cell_index: int or str, the index of the cell to plot
    ks: Sequence[int], the number of frequencies to plot
    metric: str or callable, the metric to use for evaluation
    boundary_type: str, the type of boundary to plot
    normalize_area: bool, whether to normalize the area of the boundary
    fps: int, the number of frames per second
    dpi: int, the resolution of the GIF
    save_path: str or Path, the path to save the GIF

    Returns
    -------
    Path, the path to the saved GIF
    """
    cell_id = adata.obs_names[cell_index] if isinstance(cell_index, int) else str(cell_index)
    pts_orig = normalize_boundary(
        get_cell_boundary_points(adata, cell_id, boundary_type), normalize_area
    )
    z_orig = pts_orig[:, 0] + 1j * pts_orig[:, 1]
    n = len(z_orig)

    # ks default
    if ks is None:
        ks = list(range(1, n//2))
    ks = sorted({int(k) for k in ks if 0 < k < n//2})

    # Metric selection
    if isinstance(metric, str):
        metric_fn = _METRICS.get(metric.lower())
        if metric_fn is None:
            raise ValueError(f"Unsupported metric '{metric}'")
        metric_name = metric.upper()
    else:
        metric_fn = metric
        metric_name = getattr(metric, "__name__", "metric")

    # Setup figure
    fig, (ax_shape, ax_curve) = plt.subplots(
        1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 2]}
    )
    orig_line, = ax_shape.plot(pts_orig[:, 0], pts_orig[:, 1], lw=1, color="lightgray")
    recon_line, = ax_shape.plot([], [], lw=2, color="tab:blue")
    ax_shape.set_aspect("equal"); ax_shape.axis("off")
    ax_curve.set_xlabel("k"); ax_curve.set_ylabel(metric_name); ax_curve.grid(True)

    errors: list[float] = []
    err_line, = ax_curve.plot([], [], marker="o")
    current_pt = ax_curve.scatter([], [], color="red", zorder=5)

    def init():
        recon_line.set_data([], []);
        err_line.set_data([], []);
        current_pt.set_offsets(np.empty((0,2)))
        ax_curve.set_xlim(0, max(ks)); ax_curve.set_ylim(0, 1)
        return recon_line, err_line, current_pt

    def update(i: int):
        k = ks[i]
        pts_rec = reconstruct_from_fourier(z_orig, k)
        recon_line.set_data(pts_rec[:,0], pts_rec[:,1])
        err = metric_fn(pts_orig, pts_rec)
        errors.append(err)
        err_line.set_data(ks[:i+1], errors)
        current_pt.set_offsets([[k, err]])
        ax_shape.set_title(f"Cell {cell_id} — k={k} ({metric_name}={err:.4f})")
        if i == 0:
            ax_curve.set_ylim(0, max(errors)*1.05)
        return recon_line, err_line, current_pt

    anim = animation.FuncAnimation(
        fig, update, frames=len(ks), init_func=init,
        interval=1000/fps, blit=False, repeat=False
    )
    out_path = Path(save_path).with_suffix(".gif")
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"GIF saved ➜ {out_path.resolve()}")
    return out_path