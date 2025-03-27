# tic/data/microe.py

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import anndata
from torch_geometric.data import Data

from tic.data.cell import Cell
from tic.constant import ALL_BIOMARKERS, ALL_CELL_TYPES, DEFAULT_REPRESENTATION_PIPELINE, REPRESENTATION_METHODS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MicroE:
    """
    Represents the microenvironment centered on a specific cell.
    
    Core Attributes:
      - center_cell: The primary Cell object.
      - neighbors: A list of Cell objects representing neighboring cells.
      - cells: A unified list combining the center cell and its neighbors.
      - tissue_id: Identifier for the tissue or region.
      - graph: An optional PyTorch Geometric Data object representing the graph of the microenvironment.
    
    Core Functionalities:
      - Conversion of the microenvironment into a graph via a provided node and edge feature extraction functions.
      - Computation of aggregated features (e.g., neighbor biomarker matrix).
      - Computation of various representation vectors (raw expression, neighbor composition, NN embedding).
      - Exporting the center cell with attached representations.
      - Conversion to AnnData for further downstream analysis.
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
        
        :param center_cell: The central Cell object.
        :param neighbors: A list of neighboring Cell objects.
        :param tissue_id: Tissue identifier.
        :param graph: Optional precomputed PyG graph (Data object).
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
        Convert the microenvironment into a PyG graph.
        
        This method uses lazy evaluation. If a graph is already cached, it returns it directly.
        
        :param node_feature_fn: A function that extracts node features from a Cell.
        :param edge_index_fn: A function that computes the edge indices for a list of Cells.
        :param edge_attr_fn: Optional function that computes edge attributes given two Cells.
        :return: A PyG Data object representing the microenvironment graph.
        :raises ValueError: If node features cannot be generated.
        """
        if self.graph is not None:
            return self.graph
        
        try:
            node_features = torch.tensor(
                [node_feature_fn(cell) for cell in self.cells], dtype=torch.float
            )
        except Exception as e:
            raise ValueError(f"Error generating node features: {e}")
        
        edge_index: torch.Tensor = edge_index_fn(self.cells)
        edge_attr: Optional[torch.Tensor] = None
        if edge_attr_fn is not None:
            edges = edge_index.t().tolist()
            try:
                edge_attr = torch.tensor(
                    [edge_attr_fn(self.cells[i], self.cells[j]) for i, j in edges],
                    dtype=torch.float
                )
            except Exception as e:
                logger.warning(f"Error generating edge attributes: {e}")
                edge_attr = None
        
        self.graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        return self.graph

    def get_neighborhood_biomarker_matrix(
        self,
        biomarkers: List[str] = ALL_BIOMARKERS,
        cell_types: List[str] = ALL_CELL_TYPES
    ) -> np.ndarray:
        """
        Generate a biomarker expression matrix for the neighbor cells (excluding the center cell).
        
        Rows correspond to each cell type (as provided) and columns to each biomarker.
        Missing values are filled with np.nan.
        
        :param biomarkers: List of biomarker names.
        :param cell_types: List of cell types.
        :return: A 2D numpy array with averaged biomarker expressions for each cell type.
        """
        neighbor_cells: List[Cell] = self.neighbors
        biomarker_matrix: np.ndarray = np.full((len(cell_types), len(biomarkers)), np.nan, dtype=float)
        for i, ctype in enumerate(cell_types):
            cells_of_type = [cell for cell in neighbor_cells if cell.cell_type == ctype]
            if cells_of_type:
                for j, biomarker in enumerate(biomarkers):
                    values = [cell.get_biomarker(biomarker) for cell in cells_of_type if cell.get_biomarker(biomarker) is not None]
                    if values:
                        biomarker_matrix[i, j] = np.mean(values)
        return biomarker_matrix

    def _get_raw_expression(self, biomarkers: List[str] = ALL_BIOMARKERS) -> np.ndarray:
        """
        Compute a raw expression vector using the center cell's biomarker data.
        
        :param biomarkers: List of biomarker names.
        :return: A numpy array of biomarker expression values.
        """
        return np.array([self.center_cell.get_biomarker(bm) for bm in biomarkers], dtype=float)

    def _get_neighbor_composition(self, cell_types: List[str] = ALL_CELL_TYPES) -> np.ndarray:
        """
        Compute the neighbor cell composition as fractions of each cell type.
        
        :param cell_types: List of cell types.
        :return: A numpy array representing the fraction of neighbors for each cell type.
        """
        total_neighbors: int = len(self.neighbors)
        counts: List[float] = []
        for ctype in cell_types:
            n: int = sum(1 for c in self.neighbors if c.cell_type == ctype)
            counts.append(n / total_neighbors if total_neighbors > 0 else 0.0)
        return np.array(counts, dtype=float)

    def _get_nn_embedding(self, model: nn.Module, device: torch.device) -> np.ndarray:
        """
        Compute an embedding for the microenvironment using a provided neural network.
        
        :param model: A PyTorch neural network model.
        :param device: The torch.device to run the model on.
        :return: A numpy array representing the computed embedding.
        :raises ValueError: If the graph is not available.
        """
        if self.graph is None:
            raise ValueError("No PyG graph found. Build or assign self.graph first.")
        model.eval()
        graph_on_device = self.graph.to(device)
        with torch.no_grad():
            embedding = model(graph_on_device)
        return embedding.cpu().numpy()

    # Mapping from representation method names to internal functions.
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
        Compute selected representations for the microenvironment and attach them as additional features
        to the center cell. Returns the updated center cell.
        
        :param representations: List of representation method names.
        :param model: Optional neural network model (required for 'nn_embedding').
        :param device: Optional torch.device (required for 'nn_embedding').
        :param biomarkers: List of biomarkers (used for raw expression).
        :param cell_types: List of cell types (used for neighbor composition).
        :return: The center Cell with new features attached.
        """
        if representations is None:
            representations = DEFAULT_REPRESENTATION_PIPELINE
        
        for method_name in representations:
            func = self._REPRESENTATION_FUNCS.get(method_name)
            if func is None:
                logger.warning(f"Unknown representation method '{method_name}' - skipping.")
                continue
            
            # Call the appropriate function with the needed parameters.
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
            
            # Attach the representation vector as an additional feature to the center cell.
            self.center_cell.add_feature(method_name, rep_vec)
        
        return self.center_cell

    def get_center_biomarker_vector(self, y_biomarkers: Any) -> np.ndarray:
        """
        Retrieve one or multiple biomarker expression values from the center cell.
        
        :param y_biomarkers: A biomarker name or a list of names.
        :return: A numpy array of the biomarker expression values.
        """
        if isinstance(y_biomarkers, str):
            y_biomarkers = [y_biomarkers]
        return np.array([self.center_cell.get_biomarker(bm) for bm in y_biomarkers], dtype=float)

    def prepare_for_causal_inference(
        self,
        y_biomarkers: Any,
        x_biomarkers: List[str] = ALL_BIOMARKERS,
        x_cell_types: List[str] = ALL_CELL_TYPES
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Prepare and return data for causal inference.
        
        :param y_biomarkers: Biomarker(s) to be used as outcome variables.
        :param x_biomarkers: Biomarker names for the predictor variables.
        :param x_cell_types: Cell types for generating predictors.
        :return: A tuple (X, Y, X_labels, Y_labels) where:
                 - X is a flattened predictor vector,
                 - Y is a vector of center cell biomarkers,
                 - X_labels and Y_labels are the corresponding feature labels.
        """
        neighborhood_matrix: np.ndarray = self.get_neighborhood_biomarker_matrix(biomarkers=x_biomarkers, cell_types=x_cell_types)
        X: np.ndarray = neighborhood_matrix.flatten()
        X_labels: List[str] = [f"{bm}&{ct}" for ct in x_cell_types for bm in x_biomarkers]
        Y: np.ndarray = self.get_center_biomarker_vector(y_biomarkers)
        Y_labels: List[str] = [y_biomarkers] if isinstance(y_biomarkers, str) else y_biomarkers
        return X, Y, X_labels, Y_labels

    def to_anndata(self) -> anndata.AnnData:
        """
        Convert the entire microenvironment (center cell + neighbors) into an AnnData object.
        
        The AnnData object contains:
          - X: a biomarker expression matrix,
          - obs: cell metadata,
          - obsm["spatial"]: spatial coordinates,
          - var: biomarker annotations.
        
        :return: anndata.AnnData object representing the microenvironment.
        """
        all_biomarkers = set()
        for cell in self.cells:
            all_biomarkers.update(cell.biomarkers.biomarkers.keys())
        biomarker_names: List[str] = sorted(all_biomarkers)
        
        X_list: List[List[float]] = []
        obs_data: List[Dict[str, Any]] = []
        index: List[str] = []
        for cell in self.cells:
            expr = [cell.biomarkers.biomarkers.get(bm, np.nan) for bm in biomarker_names]
            X_list.append(expr)
            meta: Dict[str, Any] = {"CELL_TYPE": cell.cell_type, "SIZE": cell.size}
            meta.update(cell.additional_features)
            obs_data.append(meta)
            index.append(cell.cell_id)
        
        X_arr = np.array(X_list)
        obs_df = pd.DataFrame(obs_data, index=index)
        var_df = pd.DataFrame(index=biomarker_names)
        spatial_coords = np.array([cell.pos for cell in self.cells])
        obsm: Dict[str, Any] = {"spatial": spatial_coords}
        
        adata = anndata.AnnData(X=X_arr, obs=obs_df, var=var_df, obsm=obsm)
        adata.uns["data_level"] = "microe"
        adata.uns["center_cell_id"] = self.center_cell.cell_id
        adata.uns["tissue_id"] = self.tissue_id
        return adata

    def __str__(self) -> str:
        """Return a string summary of the microenvironment."""
        return f"Microenvironment around Cell {self.center_cell.cell_id} with {len(self.neighbors)} neighbors"