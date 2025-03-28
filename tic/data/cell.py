# tic/data/cell

from typing import Any, Dict, Optional, Tuple
import numpy as np
import anndata

from tic.data.utils import build_ann_data

class Biomarkers:
    """
    A class to manage biomarkers and their expression levels.
    
    Biomarkers are stored as a dictionary mapping biomarker names (str)
    to their expression levels (float).
    """
    def __init__(self, **biomarker_values: float) -> None:
        """
        Initializes the Biomarkers object with dynamic biomarker data.
        """
        self.biomarkers: Dict[str, float] = biomarker_values

    def __getattr__(self, biomarker_name: str) -> float:
        """
        Retrieve the expression level of a specified biomarker.
        
        :param biomarker_name: The name of the biomarker to retrieve.
        :return: The expression level if it exists.
        :raises AttributeError: If the biomarker is not found.
        """
        biomarker_dict = self.__dict__.get("biomarkers", {})
        if biomarker_name in biomarker_dict:
            return biomarker_dict[biomarker_name]
        else:
            raise AttributeError(f"Biomarker '{biomarker_name}' not found in this cell.")

    def __repr__(self) -> str:
        """Returns a string representation of the biomarkers."""
        return f"Biomarkers({self.biomarkers})"

    def __getitem__(self, key: str) -> float:
        """Allow dictionary-like access."""
        return self.__getattr__(key)

class Cell:
    """
    A class representing a single cell with attributes such as position, size, biomarkers, etc.
    """
    def __init__(
        self,
        tissue_id: str,
        cell_id: str,
        pos: Tuple[float, ...],
        size: float,
        cell_type: Optional[str] = None,
        biomarkers: Optional[Biomarkers] = None,
        **additional_features: Any
    ) -> None:
        """
        Initializes the Cell object.
        
        :param tissue_id: Tissue identifier.
        :param cell_id: Unique cell identifier.
        :param pos: The cell's spatial position (e.g., (x, y) or (x, y, z)).
        :param size: The cell's size or volume.
        :param cell_type: Type of the cell.
        :param biomarkers: A Biomarkers object; if None, defaults to an empty Biomarkers.
        :param additional_features: Additional cell features.
        """
        self.tissue_id: str = tissue_id
        self.cell_id: str = cell_id
        self.pos: Tuple[float, ...] = pos
        self.size: float = size
        self.cell_type: Optional[str] = cell_type
        self.biomarkers: Biomarkers = biomarkers if biomarkers is not None else Biomarkers()
        self.additional_features: Dict[str, Any] = additional_features

    def __str__(self) -> str:
        return f"Cell {self.cell_id} at position {self.pos} with size {self.size} and type {self.cell_type}"
    
    def get_biomarker(self, biomarker_name: str) -> Optional[float]:
        """
        Retrieves the expression level of a specific biomarker.
        
        :param biomarker_name: Name of the biomarker.
        :return: Expression level if exists; otherwise, returns None.
        """
        try:
            return self.biomarkers.__getattr__(biomarker_name)
        except AttributeError:
            print(f"Warning: Biomarker '{biomarker_name}' not found in cell {self.cell_id}.")
            return None
    
    def add_feature(self, feature_name: str, feature_value: Any) -> None:
        """
        Adds or updates an additional feature.
        """
        self.additional_features[feature_name] = feature_value
    
    def get_feature(self, feature_name: str) -> Optional[Any]:
        """
        Retrieves the value of a specific additional feature.
        """
        return self.additional_features.get(feature_name, None)
    
    def to_anndata(self) -> anndata.AnnData:
        """
        Convert this Cell object into an AnnData object with a standardized format.
        
        The output AnnData follows these conventions:
          - X: A biomarker expression matrix (1 x n_biomarkers). If no biomarkers are present, X is empty.
          - obs: A DataFrame containing cell metadata including:
                   - cell_type, size, and any additional features.
                   The index is set to the cell_id.
          - var: A DataFrame whose index consists of biomarker names.
          - obsm["spatial"]: A NumPy array storing the cell's spatial coordinates.
          - uns: A dictionary containing {"data_level": "cell"}.
        
        :return: anndata.AnnData object representing this cell.
        """
        # Construct X from biomarkers.
        biomarker_names = list(self.biomarkers.biomarkers.keys())
        if biomarker_names:
            X = np.array([[self.biomarkers.biomarkers.get(bm, np.nan) for bm in biomarker_names]])
        else:
            X = np.empty((1, 0))
        
        # Build obs metadata.
        obs_data = {
            "tissue_id": self.tissue_id,
            "cell_id": self.cell_id,
            "cell_type": self.cell_type,
            "size": self.size,
        }
        obs_data.update(self.additional_features)
        
        # Build uns metadata.
        uns = {
            "data_level": "cell",
            "tissue_id": self.tissue_id,
        }
        
        # Use the helper function to construct a standardized AnnData.
        adata = build_ann_data(
            cells=[self],
            X=X,
            extra_obs=[obs_data],
            uns=uns,
            feature_names=biomarker_names if biomarker_names else []
        )
        return adata