# tic/data/microe.py
import anndata
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Any, Callable, Dict, List, Optional, Tuple

from tic.constant import ALL_BIOMARKERS, ALL_CELL_TYPES, DEFAULT_REPRESENTATION_PIPELINE, REPRESENTATION_METHODS
from tic.data.cell import Cell

class MicroE:
    """
    Represents the microenvironment centered on a specific cell.
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
        """
        self.center_cell: Cell = center_cell
        self.neighbors: List[Cell] = neighbors
        self.cells: List[Cell] = [center_cell] + neighbors
        self.tissue_id: str = tissue_id
        self.graph: Optional[Data] = graph

    def get_center_cell(self) -> Cell:
        return self.center_cell
    
    def get_neighbors(self) -> List[Cell]:
        return self.neighbors

    def to_graph(
        self,
        node_feature_fn: Callable[[Cell], Any],
        edge_index_fn: Callable[[List[Cell]], torch.Tensor],
        edge_attr_fn: Optional[Callable[[Cell, Cell], Any]] = None
    ) -> Data:
        """
        Convert the microenvironment into a PyG graph.
        """
        if self.graph is not None:
            return self.graph
        
        node_features = torch.tensor([node_feature_fn(cell) for cell in self.cells], dtype=torch.float)
        edge_index: torch.Tensor = edge_index_fn(self.cells)
        edge_attr: Optional[torch.Tensor] = None
        if edge_attr_fn is not None:
            edges = edge_index.t().tolist()
            edge_attr = torch.tensor(
                [edge_attr_fn(self.cells[i], self.cells[j]) for i, j in edges],
                dtype=torch.float
            )
        self.graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        return self.graph

    def get_neighborhood_biomarker_matrix(
        self,
        biomarkers: List[str] = ALL_BIOMARKERS,
        cell_types: List[str] = ALL_CELL_TYPES
    ) -> np.ndarray:
        """
        Generate a biomarker expression matrix for the neighborhood (excluding the center cell).
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
        """Directly use the center cell's biomarker expression as a vector."""
        return np.array([self.center_cell.get_biomarker(bm) for bm in biomarkers], dtype=float)

    def _get_neighbor_composition(self, cell_types: List[str] = ALL_CELL_TYPES) -> np.ndarray:
        """Use neighbor composition (fraction of each cell type) as a vector."""
        total_neighbors: int = len(self.neighbors)
        counts: List[float] = []
        for ctype in cell_types:
            n: int = sum(1 for c in self.neighbors if c.cell_type == ctype)
            counts.append(n / total_neighbors if total_neighbors > 0 else 0.0)
        return np.array(counts, dtype=float)

    def _get_nn_embedding(self, model: nn.Module, device: torch.device) -> np.ndarray:
        """Use a neural network to get an embedding from the PyG graph."""
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
        Attach selected representation vectors to the center cell and return it.
        """
        if representations is None:
            representations = DEFAULT_REPRESENTATION_PIPELINE
        
        for method_name in representations:
            func = self._REPRESENTATION_FUNCS.get(method_name, None)
            if func is None:
                print(f"Warning: Unknown representation method '{method_name}' - skipping.")
                continue
            
            if method_name == REPRESENTATION_METHODS["raw_expression"]:
                rep_vec = func(self, biomarkers=biomarkers)
            elif method_name == REPRESENTATION_METHODS["neighbor_composition"]:
                rep_vec = func(self, cell_types=cell_types)
            elif method_name == REPRESENTATION_METHODS["nn_embedding"]:
                if model is None or device is None:
                    print("Warning: 'nn_embedding' requires model and device. Skipping.")
                    continue
                rep_vec = func(self, model=model, device=device)
            else:
                print(f"Warning: method '{method_name}' not implemented.")
                continue
            
            self.center_cell.add_feature(method_name, rep_vec)
        
        return self.center_cell

    def get_center_biomarker_vector(self, y_biomarkers: Any) -> np.ndarray:
        """
        Retrieve one or multiple biomarkers from the center cell.
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
        Prepare the data needed for causal inference.
        """
        neighborhood_matrix: np.ndarray = self.get_neighborhood_biomarker_matrix(biomarkers=x_biomarkers, cell_types=x_cell_types)
        X: np.ndarray = neighborhood_matrix.flatten()
        X_labels: List[str] = [f"{bm}&{ct}" for ct in x_cell_types for bm in x_biomarkers]
        Y: np.ndarray = self.get_center_biomarker_vector(y_biomarkers)
        Y_labels: List[str] = [y_biomarkers] if isinstance(y_biomarkers, str) else y_biomarkers
        return X, Y, X_labels, Y_labels

    def to_anndata(self) -> anndata.AnnData:
        """
        Convert this MicroE object to an AnnData object.
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
        return f"Microenvironment around Cell {self.center_cell.cell_id} with {len(self.neighbors)} neighbors"