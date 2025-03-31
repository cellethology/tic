"""
Module: tic.data.cell

Contains classes representing individual cells and their associated biomarker
data, as well as utility methods for converting cells to AnnData objects.
"""

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
        Initialize the Biomarkers object with dynamic biomarker data.

        Parameters
        ----------
        biomarker_values : float
            Keyword arguments mapping biomarker names to their expression levels.
        """
        self.biomarkers: Dict[str, float] = biomarker_values

    def __getattr__(self, biomarker_name: str) -> float:
        """
        Retrieve the expression level of a specified biomarker.

        Parameters
        ----------
        biomarker_name : str
            The name of the biomarker to retrieve.

        Returns
        -------
        float
            The expression level if it exists.

        Raises
        ------
        AttributeError
            If the biomarker is not found in this cell.
        """
        biomarker_dict = self.__dict__.get("biomarkers", {})
        if biomarker_name in biomarker_dict:
            return biomarker_dict[biomarker_name]
        msg = f"Biomarker '{biomarker_name}' not found in this cell."
        raise AttributeError(msg)

    def __repr__(self) -> str:
        """Return a string representation of the biomarkers."""
        return f"Biomarkers({self.biomarkers})"

    def __getitem__(self, key: str) -> float:
        """
        Support dictionary-like access.

        Parameters
        ----------
        key : str
            The biomarker name.

        Returns
        -------
        float
            The expression level of the biomarker.
        """
        return self.__getattr__(key)


class Cell:
    """
    A class representing a single cell with attributes such as position,
    size, biomarkers, etc.
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
        Initialize the Cell object.

        Parameters
        ----------
        tissue_id : str
            Tissue identifier.
        cell_id : str
            Unique cell identifier.
        pos : Tuple[float, ...]
            The cell's spatial position (e.g., (x, y) or (x, y, z)).
        size : float
            The cell's size or volume.
        cell_type : Optional[str]
            Type of the cell (e.g., "Tumor", "Immune").
        biomarkers : Optional[Biomarkers]
            A Biomarkers object; if None, defaults to an empty Biomarkers.
        additional_features : Any
            Additional cell features provided as keyword arguments.
        """
        self.tissue_id: str = tissue_id
        self.cell_id: str = cell_id
        self.pos: Tuple[float, ...] = pos
        self.size: float = size
        self.cell_type: Optional[str] = cell_type
        self.biomarkers: Biomarkers = (
            biomarkers if biomarkers is not None else Biomarkers()
        )
        self.additional_features: Dict[str, Any] = additional_features

    def __str__(self) -> str:
        """
        Return a string describing the cell.

        Returns
        -------
        str
            A description of cell ID, position, size, and type.
        """
        return (
            f"Cell {self.cell_id} at position {self.pos} with size "
            f"{self.size} and type {self.cell_type}"
        )

    def get_biomarker(self, biomarker_name: str) -> Optional[float]:
        """
        Retrieve the expression level of a specific biomarker.

        Parameters
        ----------
        biomarker_name : str
            Name of the biomarker.

        Returns
        -------
        Optional[float]
            Expression level if it exists, otherwise None.
        """
        try:
            return self.biomarkers.__getattr__(biomarker_name)
        except AttributeError:
            print(
                f"Warning: Biomarker '{biomarker_name}' not found in cell "
                f"{self.cell_id}."
            )
            return None

    def add_feature(self, feature_name: str, feature_value: Any) -> None:
        """
        Add or update an additional feature.

        Parameters
        ----------
        feature_name : str
            The name of the feature.
        feature_value : Any
            The value to assign to that feature.
        """
        self.additional_features[feature_name] = feature_value

    def get_feature(self, feature_name: str) -> Optional[Any]:
        """
        Retrieve the value of a specific additional feature.

        Parameters
        ----------
        feature_name : str
            The feature name.

        Returns
        -------
        Optional[Any]
            Value of the feature if it exists, otherwise None.
        """
        return self.additional_features.get(feature_name, None)

    def to_anndata(self) -> anndata.AnnData:
        """
        Convert this Cell object into an AnnData object with a standardized format.

        The output AnnData follows these conventions:
          - X: a biomarker expression matrix (1 x n_biomarkers).
            If no biomarkers are present, X is empty.
          - obs: a DataFrame containing cell metadata including:
              - cell_type, size, and any additional features.
              The index is set to the cell_id.
          - var: a DataFrame whose index consists of biomarker names.
          - obsm["spatial"]: a NumPy array storing the cell's spatial coordinates.
          - uns: a dictionary containing {"data_level": "cell"}.

        Returns
        -------
        anndata.AnnData
            An AnnData object representing this cell.
        """
        biomarker_names = list(self.biomarkers.biomarkers.keys())
        if biomarker_names:
            X = np.array(
                [
                    [
                        self.biomarkers.biomarkers.get(bm, np.nan)
                        for bm in biomarker_names
                    ]
                ]
            )
        else:
            X = np.empty((1, 0))

        obs_data = {
            "tissue_id": self.tissue_id,
            "cell_id": self.cell_id,
            "cell_type": self.cell_type,
            "size": self.size,
        }
        obs_data.update(self.additional_features)

        uns = {
            "data_level": "cell",
            "tissue_id": self.tissue_id,
        }

        adata = build_ann_data(
            cells=[self],
            X=X,
            extra_obs=[obs_data],
            uns=uns,
            feature_names=biomarker_names if biomarker_names else []
        )
        return adata