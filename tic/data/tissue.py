import numpy as np
import pandas as pd
import anndata
import torch
from torch_geometric.data import Data

from tic.constant import MICROE_NEIGHBOR_CUTOFF
from tic.data.cell import Biomarkers, Cell
from tic.data.microe import MicroE

class Tissue:
    """
    A class to represent a tissue sample, containing a list of cells, and optionally a precomputed PyG graph.
    
    The Tissue class stores all the cells within a region, provides methods to extract cell information,
    perform statistical analysis, and convert the tissue into a graph structure for use with PyTorch Geometric.
    
    This version supports:
      - Storing and reusing a precomputed PyG graph (to avoid expensive recomputation).
      - Saving the graph to disk and loading it later to instantiate a Tissue instance.
    """
    def __init__(self, tissue_id, cells, position=None, graph=None):
        """
        Initialize a Tissue object.
        
        :param tissue_id: The unique identifier for the tissue sample.
        :param cells: A list of Cell objects that make up the tissue.
        :param position: The spatial position of the tissue. Defaults to (0, 0).
        :param graph: Optional precomputed PyG graph (Data object). Defaults to None.
        """
        self.tissue_id = tissue_id
        self.cells = cells  # List of Cell objects
        self.cell_dict = {cell.cell_id: cell for cell in cells}  # O(1) lookup for cells by ID
        self.pos = position if position else (0, 0)

        # Precompute positions array for all cells (for vectorized computations)
        self.positions = np.array([cell.pos for cell in self.cells])
        self.validate_cells_positions()

        # Placeholder for the PyG graph. If provided, reuse it.
        self.graph = graph

    def validate_cells_positions(self):
        """
        Ensures that all cells in the tissue have the same number of dimensions in their position data.
        """
        if self.positions.ndim != 2:
            raise ValueError("Cell positions should be a 2D array.")
        
        expected_dim = self.positions.shape[1]
        if not all(len(cell.pos) == expected_dim for cell in self.cells):
            raise ValueError("Inconsistent position dimensions in cell data.")
        
    def get_cell_by_id(self, cell_id):
        """
        Retrieve a cell by its unique ID.
        
        :param cell_id: The unique identifier for the cell.
        :return: The Cell object, or None if not found.
        """
        return self.cell_dict.get(cell_id, None)
    
    def get_microenvironment(self, center_cell_id: str, k: int = 3, microe_neighbor_cutoff: float = MICROE_NEIGHBOR_CUTOFF) -> MicroE:
        """
        Extract the k-hop microenvironment for the specified center cell using the precomputed Tissue graph.
        
        This function:
        1. Extracts the k-hop subgraph from the Tissue graph.
        2. Reorders (reindexes) the subgraph so that the center cell becomes index 0.
        3. Ensures that the ordering of microE.cells is consistent with the node ordering in microE.graph.
        4. Filters the subgraph so that only nodes (cells) within microe_neighbor_cutoff from the center cell are retained.
        
        :param center_cell_id: The unique identifier for the center cell.
        :param k: The number of hops (k-hop subgraph) to include in the microenvironment.
        :param microe_neighbor_cutoff: The distance threshold for filtering neighboring cells.
        :return: A MicroE object representing the microenvironment, with graph, cells, and neighbors updated.
        :raises ValueError: If the Tissue graph is not computed or the center cell is not found.
        """
        from torch_geometric.utils import k_hop_subgraph, subgraph
        import torch

        if self.graph is None:
            raise ValueError("Tissue graph has not been computed. Please call to_graph() first.")

        # Find the center cell's index in self.cells.
        center_index = None
        for idx, cell in enumerate(self.cells):
            if cell.cell_id == center_cell_id:
                center_index = idx
                break
        if center_index is None:
            raise ValueError("Center cell not found in Tissue.")

        # Extract the k-hop subgraph using the Tissue's edge_index.
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            center_index, num_hops=k, edge_index=self.graph.edge_index,
            relabel_nodes=True, num_nodes=len(self.cells)
        )

        # Get edge attributes if available.
        if hasattr(self.graph, 'edge_attr') and self.graph.edge_attr is not None:
            sub_edge_attr = self.graph.edge_attr[edge_mask]
        else:
            sub_edge_attr = None

        # Get the list of cells corresponding to the nodes in the subgraph.
        micro_cells = [self.cells[i] for i in subset.tolist()]

        # Determine the center cell's index in the subgraph from mapping.
        center_sub_idx = mapping.item() if torch.is_tensor(mapping) else mapping

        # ----- Reindex: Move the center node to index 0 -----
        n = len(micro_cells)
        perm = [center_sub_idx] + list(range(0, center_sub_idx)) + list(range(center_sub_idx + 1, n))
        perm_tensor = torch.tensor(perm, dtype=torch.long)

        old_x = self.graph.x[subset]  # Original node features of the subgraph.
        new_x = old_x[perm_tensor]

        inv_perm = torch.argsort(perm_tensor)
        new_edge_index = inv_perm[sub_edge_index]

        new_edge_attr = sub_edge_attr

        micro_cells = [micro_cells[i] for i in perm]

        center_cell = micro_cells[0]

        micro_graph = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)

        micro_env = MicroE(center_cell, micro_cells, tissue_id=self.tissue_id, graph=None)
        micro_env.graph = micro_graph

        filtered_indices = []
        for i, cell in enumerate(micro_cells):
            dist = np.linalg.norm(np.array(cell.pos) - np.array(center_cell.pos))
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

    def get_biomarkers_of_all_cells(self, biomarker_name):
        """
        Retrieve the expression levels of a specified biomarker for all cells in the tissue.
        
        :param biomarker_name: The name of the biomarker (e.g., 'PanCK').
        :return: A dictionary with cell IDs as keys and the biomarker expression levels as values.
        """
        return {cell.cell_id: cell.get_biomarker(biomarker_name) for cell in self.cells}

    def get_statistics_for_biomarker(self, biomarker_name):
        """
        Calculate statistical information for a biomarker across all cells.
        
        :param biomarker_name: The name of the biomarker to analyze.
        :return: A tuple containing the mean and standard deviation of the biomarker expression levels.
        """
        biomarker_values = [cell.get_biomarker(biomarker_name) for cell in self.cells 
                            if cell.get_biomarker(biomarker_name) is not None]
        
        if not biomarker_values:
            raise ValueError(f"No data available for biomarker '{biomarker_name}'.")
        
        return np.mean(biomarker_values), np.std(biomarker_values)

    def to_graph(self, node_feature_fn: callable, edge_index_fn: callable, edge_attr_fn=None):
        """
        Convert the tissue into a graph structure compatible with PyTorch Geometric (PyG).
        
        If a precomputed graph is already stored in this Tissue instance, it will be returned directly.
        Otherwise, the graph is computed, stored, and returned.
        
        :param node_feature_fn: A function that takes a Cell object and returns a list of features for the node.
        :param edge_index_fn: A function that takes a list of Cell objects and returns an edge index tensor.
        :param edge_attr_fn: An optional function that takes two Cell objects and returns the edge attribute.
        :return: A PyG Data object representing the tissue as a graph.
        """
        if self.graph is not None:
            return self.graph

        node_features = self._generate_node_features(node_feature_fn)
        edge_index = edge_index_fn(self.cells)
        edge_attr = self._generate_edge_attributes(edge_attr_fn, edge_index)
        self.graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        return self.graph

    def _generate_node_features(self, node_feature_fn):
        try:
            node_features = torch.tensor([node_feature_fn(cell) for cell in self.cells], dtype=torch.float)
            return node_features
        except Exception as e:
            raise ValueError(f"Error generating node features: {e}")

    def _generate_edge_attributes(self, edge_attr_fn, edge_index):
        if edge_attr_fn is None:
            return None
        
        edge_attr = []
        for edge in edge_index.t().tolist():
            cell1_index, cell2_index = edge
            cell1 = self.cells[cell1_index]
            cell2 = self.cells[cell2_index]
            if cell1 and cell2:
                edge_attr.append(edge_attr_fn(cell1, cell2))
            else:
                raise ValueError(f"Error retrieving cells for edge: {edge}")
        
        return torch.tensor(edge_attr, dtype=torch.float)

    def save_graph(self, filepath: str):
        if self.graph is None:
            raise ValueError("No precomputed graph available to save. Call to_graph() first.")
        torch.save(self.graph, filepath)

    @classmethod
    def load_graph(cls, tissue_id: str, cells: list, filepath: str, position=None):
        graph = torch.load(filepath)
        return cls.from_pyg_graph(tissue_id, cells, graph, position)

    @classmethod
    def from_pyg_graph(cls, tissue_id: str, cells: list, pyg_graph: Data, position=None):
        return cls(tissue_id, cells, position, graph=pyg_graph)

    @classmethod
    def from_anndata(cls, adata: anndata.AnnData, tissue_id: str = None, position=None) -> "Tissue":
        """
        Construct a Tissue object from an AnnData object.
        
        Assumes:
          - The AnnData.obs index corresponds to cell_id and includes 'CELL_TYPE' and 'SIZE' columns.
          - obsm["spatial"] contains the spatial coordinates for each cell.
          - X is the expression matrix with columns corresponding to biomarkers, and var.index holds biomarker names.
        
        :param adata: Tissue-level AnnData object.
        :param tissue_id: Optional tissue ID. If not provided, attempts to retrieve from adata.uns.
        :param position: Optional tissue spatial position.
        :return: A Tissue object.
        """
        cells = []
        if tissue_id is None:
            tissue_id = adata.uns.get("region_id", None)
        
        if "spatial" not in adata.obsm:
            raise ValueError("AnnData.obsm missing 'spatial' key; cannot obtain spatial coordinates.")
        
        spatial_coords = adata.obsm["spatial"]
        if hasattr(adata.X, "toarray"):
            X_dense = adata.X.toarray()
        else:
            X_dense = np.array(adata.X)
        
        for i, (cell_id, row) in enumerate(adata.obs.iterrows()):
            cell_id = str(cell_id)
            pos = spatial_coords[i]
            cell_type = row.get("CELL_TYPE", None)
            size = row.get("SIZE", None)
            biomarker_dict = {}
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
        Convert this Tissue object to an AnnData object.
        
        Extracts:
          - X: Biomarker expression matrix (rows: cells, columns: biomarkers)
          - obs: Cell metadata (including cell type, size, and any additional features)
          - obsm["spatial"]: Spatial coordinates of cells (from self.positions)
          - var: Biomarker annotation (biomarker names as index)
        
        In uns, stores:
          - "data_level": "microe"
          - "tissue_id": The tissue ID.
        
        :return: An AnnData object representing the tissue.
        """
        
        # Determine the union of all biomarker names across cells.
        all_biomarkers = set()
        for cell in self.cells:
            all_biomarkers.update(cell.biomarkers.biomarkers.keys())
        biomarker_names = sorted(all_biomarkers)
        
        X_list = []
        obs_data = []
        index = []
        for cell in self.cells:
            # Build expression vector for this cell.
            expr = [cell.biomarkers.biomarkers.get(bm, np.nan) for bm in biomarker_names]
            X_list.append(expr)
            # Construct metadata dictionary.
            meta = {"CELL_TYPE": cell.cell_type, "SIZE": cell.size}
            meta.update(cell.additional_features)
            obs_data.append(meta)
            index.append(cell.cell_id)
        
        X = np.array(X_list)
        obs_df = pd.DataFrame(obs_data, index=index)
        var_df = pd.DataFrame(index=biomarker_names)
        obsm = {"spatial": self.positions}
        
        adata = anndata.AnnData(X=X, obs=obs_df, var=var_df, obsm=obsm)
        if self.tissue_id is not None:
            adata.uns["region_id"] = self.tissue_id
        adata.uns["data_level"] = "tissue"
        return adata

    def __str__(self):
        """
        Return a string representation of the Tissue object, summarizing the tissue ID and number of cells.
        """
        return f"Tissue {self.tissue_id} with {len(self.cells)} cells"