import matplotlib.pyplot as plt
from anndata import AnnData


def plot_cell_boundary(
    adata: AnnData,
    cell_index: int | str,
    ax: plt.Axes | None = None,
    boundary_type: str = "cell",
    show: bool = True,
    **kwargs
) -> plt.Axes:
    """
    Plot the polygon of a specific cell from adata.uns['cell_boundaries'].

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with .uns['cell_boundaries'] containing 'cell' or 'nucleus'.
    cell_index : int or str
        Index (int) into adata.obs OR a cell ID (str) from cell_boundaries['cell_id'].
    ax : matplotlib Axes, optional
        If given, will plot on this axes.
    boundary_type : str, default 'cell'
        Either 'cell' or 'nucleus', depending on which shape to plot.
    show : bool, default True
        Whether to call plt.show().
    kwargs : additional arguments to pass to plt.plot (e.g., color='red')

    Returns
    -------
    ax : matplotlib Axes
    """
    if boundary_type not in ("cell", "nucleus"):
        raise ValueError("boundary_type must be 'cell' or 'nucleus'")

    boundary_df = adata.uns["cell_boundaries"][boundary_type]

    # Get cell_id (as str)
    if isinstance(cell_index, int):
        cell_id = adata.obs_names[cell_index]
    else:
        cell_id = str(cell_index)

    # Extract polygon points
    cell_polygon = boundary_df[boundary_df["cell_id"] == cell_id]
    if cell_polygon.empty:
        raise ValueError(f"No boundary found for cell_id '{cell_id}'")

    x = cell_polygon["vertex_x"].values
    y = cell_polygon["vertex_y"].values

    # Close the polygon if needed
    if (x[0] != x[-1]) or (y[0] != y[-1]):
        x = list(x) + [x[0]]
        y = list(y) + [y[0]]

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(x, y, **kwargs)
    ax.set_title(f"Cell ID: {cell_id}")
    ax.set_aspect("equal")
    ax.axis("off")

    if show:
        plt.show()

    return ax