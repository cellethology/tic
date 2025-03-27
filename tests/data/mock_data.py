# tests/data/mock_data.py

from tic.data.cell import Biomarkers, Cell
from tic.data.microe import MicroE
from tic.data.tissue import Tissue

def create_mock_biomarkers(include_defaults=True, **kwargs):
    """
    Create a Biomarkers object with default values if desired.
    
    Parameters:
        include_defaults (bool): If True, include default biomarkers.
        kwargs: Additional biomarker values to override or add.
    """
    defaults = {"PanCK": 1.0, "CD3": 2.0} if include_defaults else {}
    defaults.update(kwargs)
    return Biomarkers(**defaults)

def create_mock_cell(cell_id="C1", tissue_id="T1", pos=(0, 0), size=10, cell_type="TypeA", biomarkers=None, **additional_features):
    """
    Create a mock Cell object with default parameters.
    
    Parameters:
        cell_id (str): Unique cell identifier.
        tissue_id (str): Tissue identifier.
        pos (tuple): Cell position.
        size (int/float): Cell size.
        cell_type (str): Type of the cell.
        biomarkers (Biomarkers): Biomarkers object; if None, defaults are used.
        additional_features: Any extra features.
    """
    if biomarkers is None:
        biomarkers = create_mock_biomarkers()
    return Cell(tissue_id=tissue_id, cell_id=cell_id, pos=pos, size=size, cell_type=cell_type, biomarkers=biomarkers, **additional_features)

def create_mock_microe(center_cell=None, neighbors=None, tissue_id="T1"):
    """
    Create a mock MicroE object with a center cell and its neighbors.
    
    Parameters:
        center_cell (Cell): The central cell; if None, a default one is created.
        neighbors (list): List of neighboring Cell objects; if None, defaults are created.
        tissue_id (str): Tissue identifier.
    """
    if center_cell is None:
        center_cell = create_mock_cell(cell_id="Center")
    if neighbors is None:
        neighbors = [create_mock_cell(cell_id="Neighbor1", pos=(1, 0)),
                     create_mock_cell(cell_id="Neighbor2", pos=(0, 1))]
    return MicroE(center_cell, neighbors, tissue_id=tissue_id)

def create_mock_tissue(cells=None, tissue_id="T1", position=(0, 0)):
    """
    Create a mock Tissue object with a list of cells.
    
    Parameters:
        cells (list): List of Cell objects; if None, defaults will be generated.
        tissue_id (str): Tissue identifier.
        position (tuple): Tissue position.
    """
    if cells is None:
        cells = [create_mock_cell(cell_id=f"C{i}", pos=(i, i)) for i in range(1, 4)]
    return Tissue(tissue_id=tissue_id, cells=cells, position=position)