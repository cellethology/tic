"""
Module: tic.data.microe

Defines the MicroE class, which represents a microenvironment around a center
cell, its neighbors, and an optional PyG graph representation.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import anndata
from torch_geometric.data import Data

from tic.data.cell import Cell
from tic.constant import (
    ALL_BIOMARKERS,
    ALL_CELL_TYPES,
    DEFAULT_REPRESENTATION_PIPELINE,
    REPRESENTATION_METHODS
)
from tic.data.utils import build_ann_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MicroE:
    """
    Represents the microenvironment centered on a specific cell.

    Core Attributes
    --------------
    center_cell : Cell
        The primary Cell object.
    neighbors : List[Cell]
        Neighboring Cell objects.
    cells : List[Cell]
        A combined list of the center cell and its neighbors.
    tissue_id : str
        Identifier for the tissue or region.
    graph : Optional[Data]
        A PyTorch Geometric Data object representing the microenvironment graph.

    Core Functionalities
    --------------------
    - Convert the microenvironment into a graph via node/edge feature extraction.
    - Compute aggregated features (e.g., neighbor biomarker matrix).
    - Compute representation vectors (raw expression, neighbor composition, NN embedding).
    - Export the center cell with attached representations.
    - Convert to AnnData for further downstream analysis.
    """

    def __init__(
        self,
        center_cell: Cell,
        neighbors: List[Cell],
        tissue_id: str,
        graph: Optional[Data] = None
    ) -> None:
        """
        Initialize a MicroE object.

        Parameters
        ----------
        center_cell : Cell
            The primary Cell object.
        neighbors : List[Cell]
            Neighboring Cell objects.
        tissue_id : str
            Tissue identifier.
        graph : Optional[Data]
            Optional precomputed PyG graph (Data object).
        """
        self.center_cell: Cell = center_cell
        self.neighbors: List[Cell] = neighbors
        self.cells: List[Cell] = [center_cell] + neighbors
        self.tissue_id: str = tissue_id
        self.graph: Optional[Data] = graph

    def get_center_cell(self) -> Cell:
        """Return the center cell."""
        return self.center_cell

    def get_neighbors(self) -> List[Cell]:
        """Return the list of neighbor cells."""
        return self.neighbors

    def to_graph(
        self,
        node_feature_fn: Callable[[Cell], Any],
        edge_index_fn: Callable[[List[Cell]], torch.Tensor],
        edge_attr_fn: Optional[Callable[[Cell, Cell], Any]] = None
    ) -> Data:
        """
        Convert the microenvironment into a PyG graph (lazy evaluation).

        Parameters
        ----------
        node_feature_fn : Callable[[Cell], Any]
            A function that extracts node features from a Cell.
        edge_index_fn : Callable[[List[Cell]], torch.Tensor]
            A function that computes the edge indices for a list of Cells.
        edge_attr_fn : Optional[Callable[[Cell, Cell], Any]]
            A function that computes edge attributes given two Cells.

        Returns
        -------
        Data
            A PyTorch Geometric Data object representing the microenvironment graph.

        Raises
        ------
        ValueError
            If node features cannot be generated.
        """
        if self.graph is not None:
            return self.graph

        try:
            node_features = torch.tensor(
                [node_feature_fn(cell) for cell in self.cells],
                dtype=torch.float
            )
        except Exception as exc:
            msg = f"Error generating node features: {exc}"
            raise ValueError(msg)

        edge_index: torch.Tensor = edge_index_fn(self.cells)
        edge_attr: Optional[torch.Tensor] = None

        if edge_attr_fn is not None:
            edges = edge_index.t().tolist()
            try:
                edge_attr = torch.tensor(
                    [
                        edge_attr_fn(self.cells[i], self.cells[j])
                        for i, j in edges
                    ],
                    dtype=torch.float
                )
            except Exception as exc:
                logger.warning(f"Error generating edge attributes: {exc}")
                edge_attr = None

        self.graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        return self.graph

    def get_neighborhood_biomarker_matrix(
        self,
        biomarkers: List[str] = ALL_BIOMARKERS,
        cell_types: List[str] = ALL_CELL_TYPES
    ) -> np.ndarray:
        """
        Generate a biomarker expression matrix for the neighbor cells
        (excluding the center cell).

        Rows correspond to each cell type (from cell_types) and columns
        to each biomarker. Missing values are filled with np.nan.

        Parameters
        ----------
        biomarkers : List[str]
            List of biomarker names.
        cell_types : List[str]
            List of cell types.

        Returns
        -------
        np.ndarray
            A 2D numpy array with averaged biomarker expressions for
            each cell type and biomarker.
        """
        neighbor_cells: List[Cell] = self.neighbors
        biomatrix = np.full(
            (len(cell_types), len(biomarkers)),
            np.nan,
            dtype=float
        )
        for i, ctype in enumerate(cell_types):
            cells_of_type = [
                cell for cell in neighbor_cells if cell.cell_type == ctype
            ]
            if cells_of_type:
                for j, biomarker in enumerate(biomarkers):
                    values: List[float] = []
                    for c in cells_of_type:
                        val = c.get_biomarker(biomarker)
                        if val is not None:
                            values.append(val)
                    if values:
                        biomatrix[i, j] = np.mean(values)
        return biomatrix

    def _get_raw_expression(
        self,
        biomarkers: List[str] = ALL_BIOMARKERS
    ) -> np.ndarray:
        """
        Compute a raw expression vector for the center cell.

        Parameters
        ----------
        biomarkers : List[str]
            List of biomarker names.

        Returns
        -------
        np.ndarray
            A numpy array of the center cell's biomarker expressions.
        """
        return np.array(
            [self.center_cell.get_biomarker(bm) for bm in biomarkers],
            dtype=float
        )

    def _get_neighbor_composition(
        self,
        cell_types: List[str] = ALL_CELL_TYPES
    ) -> np.ndarray:
        """
        Compute the neighbor cell composition as fractions of each cell type.

        Parameters
        ----------
        cell_types : List[str]
            List of cell types.

        Returns
        -------
        np.ndarray
            A numpy array representing the fraction of neighbors
            belonging to each cell type.
        """
        total_neighbors: int = len(self.neighbors)
        counts: List[float] = []
        for ctype in cell_types:
            count_type = sum(
                1 for c in self.neighbors if c.cell_type == ctype
            )
            fraction = count_type / total_neighbors if total_neighbors > 0 else 0.0
            counts.append(fraction)
        return np.array(counts, dtype=float)

    def _get_nn_embedding(
        self,
        model: nn.Module,
        device: torch.device
    ) -> np.ndarray:
        """
        Compute an embedding for the microenvironment using a neural network.

        Parameters
        ----------
        model : nn.Module
            A PyTorch neural network model.
        device : torch.device
            The device to run the model on.

        Returns
        -------
        np.ndarray
            A numpy array representing the computed embedding.

        Raises
        ------
        ValueError
            If the graph is not available.
        """
        if self.graph is None:
            raise ValueError("No PyG graph found. Build or assign self.graph first.")
        model.eval()
        graph_on_device = self.graph.to(device)
        with torch.no_grad():
            embedding = model(graph_on_device)
        return embedding.cpu().numpy()

    _REPRESENTATION_FUNCS: Dict[str, Any] = {
        REPRESENTATION_METHODS["raw_expression"]: _get_raw_expression,
        REPRESENTATION_METHODS["neighbor_composition"]: _get_neighbor_composition,
        REPRESENTATION_METHODS["nn_embedding"]: _get_nn_embedding
    }

    def export_center_cell_with_representations(
        self,
        representations: Optional[List[str]] = DEFAULT_REPRESENTATION_PIPELINE,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        biomarkers: List[str] = ALL_BIOMARKERS,
        cell_types: List[str] = ALL_CELL_TYPES
    ) -> Cell:
        """
        Compute selected representations for the microenvironment and attach them
        as additional features to the center cell.

        Parameters
        ----------
        representations : Optional[List[str]]
            List of representation method names.
        model : Optional[nn.Module]
            A neural network model (required for 'nn_embedding').
        device : Optional[torch.device]
            A torch device (required for 'nn_embedding').
        biomarkers : List[str]
            Biomarkers for raw expression representation.
        cell_types : List[str]
            Cell types for neighbor composition representation.

        Returns
        -------
        Cell
            The center Cell with new representations attached as additional features.
        """
        if representations is None:
            representations = DEFAULT_REPRESENTATION_PIPELINE

        for method_name in representations:
            func = self._REPRESENTATION_FUNCS.get(method_name)
            if func is None:
                logger.warning(f"Unknown representation method '{method_name}' - skipping.")
                continue

            if method_name == REPRESENTATION_METHODS["raw_expression"]:
                rep_vec = func(self, biomarkers=biomarkers)
            elif method_name == REPRESENTATION_METHODS["neighbor_composition"]:
                rep_vec = func(self, cell_types=cell_types)
            elif method_name == REPRESENTATION_METHODS["nn_embedding"]:
                if model is None or device is None:
                    logger.warning("'nn_embedding' requires model and device. Skipping.")
                    continue
                rep_vec = func(self, model=model, device=device)
            else:
                logger.warning(f"Method '{method_name}' not implemented.")
                continue

            self.center_cell.add_feature(method_name, rep_vec)
        return self.center_cell

    def get_center_biomarker_vector(
        self,
        y_biomarkers: Any
    ) -> np.ndarray:
        """
        Retrieve one or multiple biomarker expression values from the center cell.

        Parameters
        ----------
        y_biomarkers : Any
            A biomarker name or list of biomarker names.

        Returns
        -------
        np.ndarray
            A numpy array of the biomarker expression values for the center cell.
        """
        if isinstance(y_biomarkers, str):
            y_biomarkers = [y_biomarkers]
        return np.array(
            [self.center_cell.get_biomarker(bm) for bm in y_biomarkers],
            dtype=float
        )

    def prepare_for_causal_inference(
        self,
        y_biomarkers: Any,
        x_biomarkers: List[str] = ALL_BIOMARKERS,
        x_cell_types: List[str] = ALL_CELL_TYPES
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Prepare data for causal inference.

        Parameters
        ----------
        y_biomarkers : Any
            Outcome biomarker(s).
        x_biomarkers : List[str]
            Predictor biomarker names.
        x_cell_types : List[str]
            Cell types used to generate predictors.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[str], List[str]]
            X, Y, X_labels, Y_labels
        """
        neighborhood_matrix = self.get_neighborhood_biomarker_matrix(
            biomarkers=x_biomarkers,
            cell_types=x_cell_types
        )
        X = neighborhood_matrix.flatten()
        X_labels = [
            f"{bm}&{ct}" for ct in x_cell_types for bm in x_biomarkers
        ]
        Y = self.get_center_biomarker_vector(y_biomarkers)
        if isinstance(y_biomarkers, str):
            Y_labels = [y_biomarkers]
        else:
            Y_labels = y_biomarkers
        return X, Y, X_labels, Y_labels

    def to_anndata(self) -> anndata.AnnData:
        """
        Convert the microenvironment to an AnnData object.

        Returns
        -------
        anndata.AnnData
            The resulting AnnData with data_level = 'microe'.
        """
        all_biomarkers: set[str] = set()
        for cell in self.cells:
            all_biomarkers.update(cell.biomarkers.biomarkers.keys())
        biomarker_names = sorted(all_biomarkers)

        X_list = []
        extra_obs = []
        for i, cell in enumerate(self.cells):
            expr = [
                cell.biomarkers.biomarkers.get(bm, np.nan)
                for bm in biomarker_names
            ]
            X_list.append(expr)
            meta = {
                "tissue_id": cell.tissue_id,
                "cell_id": cell.cell_id,
                "cell_type": cell.cell_type,
                "size": cell.size,
                "cell_role": "center" if i == 0 else "neighbor",
            }
            meta.update(cell.additional_features)
            extra_obs.append(meta)

        X = np.array(X_list)
        uns = {
            "data_level": "microe",
            "tissue_id": self.tissue_id,
            "center_cell_id": self.center_cell.cell_id,
        }
        return build_ann_data(
            self.cells,
            X=X,
            extra_obs=extra_obs,
            uns=uns
        )

    def __str__(self) -> str:
        """Return a string summary of the microenvironment."""
        return (
            f"Microenvironment around Cell {self.center_cell.cell_id} "
            f"with {len(self.neighbors)} neighbors"
        )