# tic/data/tissue.py
import numpy as np
import pandas as pd
import anndata
import torch
from torch_geometric.data import Data
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from tic.constant import MICROE_NEIGHBOR_CUTOFF
from tic.data.cell import Biomarkers, Cell
from tic.data.microe import MicroE

class Tissue:
    """
    A class to represent a tissue sample, containing a list of cells and an optional precomputed PyG graph.
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
        """
        self.tissue_id: str = tissue_id
        self.cells: List[Cell] = cells
        self.cell_dict: Dict[str, Cell] = {cell.cell_id: cell for cell in cells}
        self.pos: Tuple[float, float] = position if position is not None else (0, 0)
        self.positions: np.ndarray = np.array([cell.pos for cell in self.cells])
        self.validate_cells_positions()
        self.graph: Optional[Data] = graph

    def validate_cells_positions(self) -> None:
        """
        Ensures that all cells have positions of consistent dimensions.
        """
        if self.positions.ndim != 2:
            raise ValueError("Cell positions should be a 2D array.")
        expected_dim = self.positions.shape[1]
        if not all(len(cell.pos) == expected_dim for cell in self.cells):
            raise ValueError("Inconsistent position dimensions in cell data.")

    def get_cell_by_id(self, cell_id: str) -> Optional[Cell]:
        return self.cell_dict.get(cell_id, None)
    
    def get_microenvironment(self, center_cell_id: str, k: int = 3, microe_neighbor_cutoff: float = MICROE_NEIGHBOR_CUTOFF) -> MicroE:
        """
        Extract the k-hop microenvironment for the specified center cell.
        """
        from torch_geometric.utils import k_hop_subgraph, subgraph
        if self.graph is None:
            raise ValueError("Tissue graph has not been computed. Please call to_graph() first.")

        center_index: Optional[int] = None
        for idx, cell in enumerate(self.cells):
            if cell.cell_id == center_cell_id:
                center_index = idx
                break
        if center_index is None:
            raise ValueError("Center cell not found in Tissue.")

        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            center_index, num_hops=k, edge_index=self.graph.edge_index,
            relabel_nodes=True, num_nodes=len(self.cells)
        )

        sub_edge_attr = self.graph.edge_attr[edge_mask] if (hasattr(self.graph, 'edge_attr') and self.graph.edge_attr is not None) else None
        micro_cells: List[Cell] = [self.cells[i] for i in subset.tolist()]
        center_sub_idx = mapping.item() if isinstance(mapping, torch.Tensor) else mapping

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
        return {cell.cell_id: cell.get_biomarker(biomarker_name) for cell in self.cells}

    def get_statistics_for_biomarker(self, biomarker_name: str) -> Tuple[float, float]:
        biomarker_values = [cell.get_biomarker(biomarker_name) for cell in self.cells if cell.get_biomarker(biomarker_name) is not None]
        if not biomarker_values:
            raise ValueError(f"No data available for biomarker '{biomarker_name}'.")
        return float(np.mean(biomarker_values)), float(np.std(biomarker_values))

    def to_graph(
        self,
        node_feature_fn: Callable[[Cell], Any],
        edge_index_fn: Callable[[List[Cell]], torch.Tensor],
        edge_attr_fn: Optional[Callable[[Cell, Cell], Any]] = None
    ) -> Data:
        if self.graph is not None:
            return self.graph
        node_features = self._generate_node_features(node_feature_fn)
        edge_index: torch.Tensor = edge_index_fn(self.cells)
        edge_attr = self._generate_edge_attributes(edge_attr_fn, edge_index)
        self.graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        return self.graph

    def _generate_node_features(self, node_feature_fn: Callable[[Cell], Any]) -> torch.Tensor:
        try:
            return torch.tensor([node_feature_fn(cell) for cell in self.cells], dtype=torch.float)
        except Exception as e:
            raise ValueError(f"Error generating node features: {e}")

    def _generate_edge_attributes(
        self,
        edge_attr_fn: Optional[Callable[[Cell, Cell], Any]],
        edge_index: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if edge_attr_fn is None:
            return None
        edge_attr: List[Any] = []
        for edge in edge_index.t().tolist():
            cell1_index, cell2_index = edge
            cell1 = self.cells[cell1_index]
            cell2 = self.cells[cell2_index]
            if cell1 and cell2:
                edge_attr.append(edge_attr_fn(cell1, cell2))
            else:
                raise ValueError(f"Error retrieving cells for edge: {edge}")
        return torch.tensor(edge_attr, dtype=torch.float)

    def save_graph(self, filepath: str) -> None:
        if self.graph is None:
            raise ValueError("No precomputed graph available to save. Call to_graph() first.")
        torch.save(self.graph, filepath)

    @classmethod
    def load_graph(cls: Type["Tissue"], tissue_id: str, cells: List[Cell], filepath: str, position: Optional[Tuple[float, float]] = None) -> "Tissue":
        graph = torch.load(filepath)
        return cls.from_pyg_graph(tissue_id, cells, graph, position)

    @classmethod
    def from_pyg_graph(cls: Type["Tissue"], tissue_id: str, cells: List[Cell], pyg_graph: Data, position: Optional[Tuple[float, float]] = None) -> "Tissue":
        return cls(tissue_id, cells, position, graph=pyg_graph)

    @classmethod
    def from_anndata(cls: Type["Tissue"], adata: anndata.AnnData, tissue_id: Optional[str] = None, position: Optional[Tuple[float, float]] = None) -> "Tissue":
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
        return f"Tissue {self.tissue_id} with {len(self.cells)} cells"