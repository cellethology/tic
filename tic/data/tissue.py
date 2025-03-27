# tic.data.tissue

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np
import pandas as pd
import torch
import anndata
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph

from tic.data.cell import Biomarkers, Cell
from tic.data.microe import MicroE
from tic.constant import MICROE_NEIGHBOR_CUTOFF


class Tissue:
    """
    Represents a tissue sample with a collection of Cell objects and an optional precomputed graph.
    
    Attributes:
      - tissue_id: Identifier for the tissue.
      - cells: List of all Cell objects.
      - cell_dict: Dictionary mapping cell IDs to Cell objects for quick lookup.
      - pos: Tissue-level position (defaulted to (0, 0)).
      - positions: A NumPy array containing each cell's spatial coordinates.
      - graph: An optional PyG Data object representing the tissue graph.
    """
    def __init__(
        self,
        tissue_id: str,
        cells: List[Cell],
        position: Optional[Tuple[float, float]] = None,
        graph: Optional[Data] = None
    ) -> None:
        """
        Initialize a Tissue object.
        
        :param tissue_id: Unique identifier for the tissue.
        :param cells: List of Cell objects comprising the tissue.
        :param position: Tissue-level spatial position (defaults to (0, 0) if not provided).
        :param graph: Optional precomputed PyG graph.
        """
        self.tissue_id: str = tissue_id
        self.cells: List[Cell] = cells
        self.cell_dict: Dict[str, Cell] = {cell.cell_id: cell for cell in cells}
        self.pos: Tuple[float, float] = position if position is not None else (0.0, 0.0)
        self.positions: np.ndarray = np.array([cell.pos for cell in self.cells])
        self.validate_cells_positions()
        self.graph: Optional[Data] = graph

    def validate_cells_positions(self) -> None:
        """
        Ensures all cells have spatial positions of consistent dimensions.
        
        :raises ValueError: If the positions array is not 2D or cell position dimensions are inconsistent.
        """
        if self.positions.ndim != 2:
            raise ValueError("Cell positions should be a 2D array.")
        expected_dim = self.positions.shape[1]
        if not all(len(cell.pos) == expected_dim for cell in self.cells):
            raise ValueError("Inconsistent position dimensions in cell data.")

    def get_cell_by_id(self, cell_id: str) -> Optional[Cell]:
        """
        Retrieve a cell by its unique ID.
        
        :param cell_id: Cell identifier.
        :return: The Cell object if found, else None.
        """
        return self.cell_dict.get(cell_id, None)
    
    def get_microenvironment(
        self,
        center_cell_id: str,
        k: int = 3,
        microe_neighbor_cutoff: float = MICROE_NEIGHBOR_CUTOFF
    ) -> MicroE:
        """
        Extract the k-hop microenvironment for a given center cell.
        
        This function:
          1. Uses the tissue's graph to extract a k-hop subgraph.
          2. Reorders the subgraph so that the center cell is at index 0.
          3. Filters cells based on a distance cutoff.
        
        :param center_cell_id: Identifier of the center cell.
        :param k: Number of hops for the subgraph.
        :param microe_neighbor_cutoff: Distance threshold for filtering neighbor cells.
        :return: A MicroE instance representing the microenvironment.
        :raises ValueError: If the tissue graph is not computed or the center cell is not found.
        """
        if self.graph is None:
            raise ValueError("Tissue graph has not been computed. Please call to_graph() first.")
        
        # Find index of the center cell.
        center_index: Optional[int] = None
        for idx, cell in enumerate(self.cells):
            if cell.cell_id == center_cell_id:
                center_index = idx
                break
        if center_index is None:
            raise ValueError("Center cell not found in Tissue.")
        
        # Extract k-hop subgraph.
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            center_index, num_hops=k, edge_index=self.graph.edge_index,
            relabel_nodes=True, num_nodes=len(self.cells)
        )
        
        # Get edge attributes if available.
        sub_edge_attr = self.graph.edge_attr[edge_mask] if (hasattr(self.graph, 'edge_attr') and self.graph.edge_attr is not None) else None
        
        # Construct microenvironment cells from the subset indices.
        micro_cells: List[Cell] = [self.cells[i] for i in subset.tolist()]
        center_sub_idx = mapping.item() if isinstance(mapping, torch.Tensor) else mapping
        
        # Permute so that center cell is first.
        n: int = len(micro_cells)
        perm: List[int] = [center_sub_idx] + list(range(0, center_sub_idx)) + list(range(center_sub_idx + 1, n))
        perm_tensor = torch.tensor(perm, dtype=torch.long)
        old_x = self.graph.x[subset]
        new_x = old_x[perm_tensor]
        inv_perm = torch.argsort(perm_tensor)
        new_edge_index = inv_perm[sub_edge_index]
        new_edge_attr = sub_edge_attr
        micro_cells = [micro_cells[i] for i in perm]
        center_cell: Cell = micro_cells[0]
        
        micro_graph = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
        micro_env = MicroE(center_cell, micro_cells, tissue_id=self.tissue_id, graph=None)
        micro_env.graph = micro_graph

        # Filter cells that are beyond the distance cutoff.
        filtered_indices: List[int] = []
        for i, cell in enumerate(micro_cells):
            dist: float = np.linalg.norm(np.array(cell.pos) - np.array(center_cell.pos))
            if dist <= microe_neighbor_cutoff:
                filtered_indices.append(i)
        if 0 not in filtered_indices:
            filtered_indices.insert(0, 0)
        filtered_indices_tensor = torch.tensor(filtered_indices, dtype=torch.long)
        filt_edge_index, filt_edge_attr = subgraph(
            filtered_indices_tensor, micro_env.graph.edge_index, micro_env.graph.edge_attr,
            relabel_nodes=True, num_nodes=micro_env.graph.num_nodes
        )
        filt_x = micro_env.graph.x[filtered_indices_tensor]
        micro_cells = [micro_cells[i] for i in filtered_indices]
        micro_env.graph = Data(x=filt_x, edge_index=filt_edge_index, edge_attr=filt_edge_attr)
        micro_env.cells = micro_cells
        micro_env.neighbors = [cell for i, cell in enumerate(micro_cells) if i != 0]
        return micro_env

    def get_biomarkers_of_all_cells(self, biomarker_name: str) -> Dict[str, Any]:
        """
        Retrieve a dictionary mapping cell IDs to the expression level of a specified biomarker.
        
        :param biomarker_name: The biomarker to retrieve.
        :return: A dictionary {cell_id: expression_value}.
        """
        return {cell.cell_id: cell.get_biomarker(biomarker_name) for cell in self.cells}

    def get_statistics_for_biomarker(self, biomarker_name: str) -> Tuple[float, float]:
        """
        Calculate the mean and standard deviation of a given biomarker's expression across all cells.
        
        :param biomarker_name: The biomarker name.
        :return: Tuple (mean, std).
        :raises ValueError: If no valid values are found.
        """
        values = [cell.get_biomarker(biomarker_name) for cell in self.cells if cell.get_biomarker(biomarker_name) is not None]
        if not values:
            raise ValueError(f"No data available for biomarker '{biomarker_name}'.")
        return float(np.mean(values)), float(np.std(values))

    def to_graph(
        self,
        node_feature_fn: Callable[[Cell], Any],
        edge_index_fn: Callable[[List[Cell]], torch.Tensor],
        edge_attr_fn: Optional[Callable[[Cell, Cell], Any]] = None
    ) -> Data:
        """
        Convert the entire tissue into a PyG graph.
        
        Uses lazy evaluation: if the graph is already computed, returns it; otherwise, computes it.
        
        :param node_feature_fn: Function to compute node features.
        :param edge_index_fn: Function to compute edge indices.
        :param edge_attr_fn: Optional function to compute edge attributes.
        :return: PyG Data object.
        """
        if self.graph is not None:
            return self.graph
        node_features = self._generate_node_features(node_feature_fn)
        edge_index = edge_index_fn(self.cells)
        edge_attr = self._generate_edge_attributes(edge_attr_fn, edge_index)
        self.graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        return self.graph

    def _generate_node_features(self, node_feature_fn: Callable[[Cell], Any]) -> torch.Tensor:
        """
        Generate node features for each cell using the provided function.
        
        :param node_feature_fn: Function that returns features for a cell.
        :return: A torch.Tensor of node features.
        :raises ValueError: If features cannot be generated.
        """
        try:
            return torch.tensor([node_feature_fn(cell) for cell in self.cells], dtype=torch.float)
        except Exception as e:
            raise ValueError(f"Error generating node features: {e}")

    def _generate_edge_attributes(
        self,
        edge_attr_fn: Optional[Callable[[Cell, Cell], Any]],
        edge_index: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Generate edge attributes using the provided function.
        
        :param edge_attr_fn: Function that returns attributes for a pair of cells.
        :param edge_index: Tensor of edge indices.
        :return: A torch.Tensor of edge attributes or None if not provided.
        :raises ValueError: If edge attributes cannot be computed.
        """
        if edge_attr_fn is None:
            return None
        edge_attrs: List[Any] = []
        for edge in edge_index.t().tolist():
            cell1_index, cell2_index = edge
            cell1 = self.cells[cell1_index]
            cell2 = self.cells[cell2_index]
            if cell1 and cell2:
                edge_attrs.append(edge_attr_fn(cell1, cell2))
            else:
                raise ValueError(f"Error retrieving cells for edge: {edge}")
        return torch.tensor(edge_attrs, dtype=torch.float)

    def save_graph(self, filepath: str) -> None:
        """
        Save the tissue graph to disk.
        
        :param filepath: Path where the graph will be saved.
        :raises ValueError: If no graph is available.
        """
        if self.graph is None:
            raise ValueError("No precomputed graph available to save. Call to_graph() first.")
        torch.save(self.graph, filepath)

    @classmethod
    def load_graph(cls: Type["Tissue"], tissue_id: str, cells: List[Cell], filepath: str, position: Optional[Tuple[float, float]] = None) -> "Tissue":
        """
        Load a Tissue object from a saved PyG graph.
        
        :param tissue_id: Tissue identifier.
        :param cells: List of Cell objects.
        :param filepath: Path to the saved graph.
        :param position: Optional tissue-level position.
        :return: A Tissue object with the loaded graph.
        """
        graph = torch.load(filepath)
        return cls.from_pyg_graph(tissue_id, cells, graph, position)

    @classmethod
    def from_pyg_graph(cls: Type["Tissue"], tissue_id: str, cells: List[Cell], pyg_graph: Data, position: Optional[Tuple[float, float]] = None) -> "Tissue":
        """
        Construct a Tissue instance from a PyG graph.
        
        :param tissue_id: Tissue identifier.
        :param cells: List of Cell objects.
        :param pyg_graph: PyG Data object representing the tissue graph.
        :param position: Optional tissue-level position.
        :return: A Tissue instance.
        """
        return cls(tissue_id, cells, position, graph=pyg_graph)

    @classmethod
    def from_anndata(cls: Type["Tissue"], adata: anndata.AnnData, tissue_id: Optional[str] = None, position: Optional[Tuple[float, float]] = None) -> "Tissue":
        """
        Instantiate a Tissue from an AnnData object.
        
        The AnnData is expected to have:
          - obs: with 'CELL_TYPE' and 'SIZE' columns (CELL_ID as index).
          - obsm["spatial"]: spatial coordinates.
          - X: expression matrix (biomarker values).
          - var: biomarker names.
        
        :param adata: anndata.AnnData object.
        :param tissue_id: Tissue identifier (if not provided, attempts to retrieve from adata.uns).
        :param position: Optional tissue-level position.
        :return: A Tissue instance.
        :raises ValueError: If required keys are missing.
        """
        cells: List[Cell] = []
        if tissue_id is None:
            tissue_id = adata.uns.get("region_id", None)
        if "spatial" not in adata.obsm:
            raise ValueError("AnnData.obsm missing 'spatial' key; cannot obtain spatial coordinates.")
        spatial_coords = adata.obsm["spatial"]
        X_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        for i, (cell_id, row) in enumerate(adata.obs.iterrows()):
            cell_id = str(cell_id)
            pos = spatial_coords[i]
            cell_type = row.get("CELL_TYPE", None)
            size = row.get("SIZE", None)
            biomarker_dict: Dict[str, float] = {}
            for j, biomarker in enumerate(adata.var.index):
                biomarker_dict[biomarker] = X_dense[i, j]
            biomarkers_obj = Biomarkers(**biomarker_dict)
            additional_features = row.drop(labels=["CELL_TYPE", "SIZE"]).to_dict()
            cell = Cell(
                tissue_id=tissue_id,
                cell_id=cell_id,
                pos=pos,
                size=size,
                cell_type=cell_type,
                biomarkers=biomarkers_obj,
                **additional_features
            )
            cells.append(cell)
        return cls(tissue_id=tissue_id, cells=cells, position=position)

    def to_anndata(self) -> anndata.AnnData:
        """
        Convert the Tissue into an AnnData object.
        
        The AnnData object includes:
          - X: biomarker expression matrix for all cells.
          - obs: cell metadata (including cell type and size).
          - obsm["spatial"]: spatial coordinates.
          - var: biomarker annotations.
        
        :return: anndata.AnnData object.
        """
        all_biomarkers = set()
        for cell in self.cells:
            all_biomarkers.update(cell.biomarkers.biomarkers.keys())
        biomarker_names: List[str] = sorted(all_biomarkers)
        
        X_list: List[List[float]] = []
        obs_data: List[Dict[str, Any]] = []
        index: List[str] = []
        for cell in self.cells:
            expr: List[float] = [cell.biomarkers.biomarkers.get(bm, np.nan) for bm in biomarker_names]
            X_list.append(expr)
            meta: Dict[str, Any] = {"CELL_TYPE": cell.cell_type, "SIZE": cell.size}
            meta.update(cell.additional_features)
            obs_data.append(meta)
            index.append(cell.cell_id)
        
        X = np.array(X_list)
        obs_df = pd.DataFrame(obs_data, index=index)
        var_df = pd.DataFrame(index=biomarker_names)
        obsm: Dict[str, Any] = {"spatial": self.positions}
        
        adata = anndata.AnnData(X=X, obs=obs_df, var=var_df, obsm=obsm)
        if self.tissue_id is not None:
            adata.uns["region_id"] = self.tissue_id
        adata.uns["data_level"] = "tissue"
        return adata

    def __str__(self) -> str:
        """Return a string summary of the Tissue."""
        return f"Tissue {self.tissue_id} with {len(self.cells)} cells"