"""
tests/data/test_export_representations.py

This file contains unit tests for the export_center_cell_with_representations method 
of the MicroE class (located in tic.data.microe).

Test Object Functionality:
    - export_center_cell_with_representations computes selected representations for the microenvironment 
      and attaches them as additional features to the center cell.
    - Representations include:
          1. raw_expression
          2. neighbor_composition
          3. nn_embedding
    - When model and device are provided, all representations are computed.
    - When model and device are missing, the nn_embedding representation should be skipped.
    
Tested Content:
    1. That when proper inputs (model and device) are provided, the center cell’s additional_features 
       contain keys for "raw_expression", "neighbor_composition", and "nn_embedding" with expected values.
    2. That when model and device are not provided, the function logs a warning and skips computing "nn_embedding".
"""

import numpy as np
import pytest
import torch
from tic.data.cell import Cell, Biomarkers
from tic.data.microe import MicroE
from tic.constant import REPRESENTATION_METHODS
import logging

# Dummy representation functions for testing:
def dummy_raw_expression(microe: MicroE, biomarkers):
    return np.array([1.0, 2.0])

def dummy_neighbor_composition(microe: MicroE, cell_types):
    return np.array([0.5, 0.5])

def dummy_nn_embedding(microe: MicroE, model, device):
    return np.array([0.1, 0.2, 0.3])

@pytest.fixture
def dummy_representation_funcs():
    """
    Fixture returning a dictionary mapping representation method keys to dummy functions.
    """
    return {
        REPRESENTATION_METHODS["raw_expression"]: dummy_raw_expression,
        REPRESENTATION_METHODS["neighbor_composition"]: dummy_neighbor_composition,
        REPRESENTATION_METHODS["nn_embedding"]: dummy_nn_embedding,
    }

@pytest.fixture
def microe_instance(dummy_representation_funcs):
    """
    Create a mock MicroE instance with a center cell and one neighbor,
    and override its _REPRESENTATION_FUNCS with dummy representation functions.
    """
    # Create a mock center cell.
    center = Cell(
        tissue_id="T1",
        cell_id="C1",
        pos=(0.0, 0.0),
        size=10.0,
        cell_type="Tumor",
        biomarkers=Biomarkers(BM1=1.0)
    )
    # Create a neighbor cell.
    neighbor = Cell(
        tissue_id="T1",
        cell_id="N1",
        pos=(1.0, 0.0),
        size=12.0,
        cell_type="Tumor",
        biomarkers=Biomarkers(BM1=2.0)
    )
    microe = MicroE(center, [neighbor], tissue_id="T1")
    # Override the internal representation function mapping.
    microe._REPRESENTATION_FUNCS = dummy_representation_funcs
    return microe

def test_export_representations_all(microe_instance):
    """
    Test export_center_cell_with_representations when model and device are provided.
    
    Expected:
      - The center cell's additional_features should contain keys for all representations:
        "raw_expression", "neighbor_composition", and "nn_embedding".
      - Their computed values should match the dummy functions.
    """
    # Provide dummy model and device.
    dummy_model = lambda x: x  # A dummy model (identity function)
    dummy_device = torch.device("cpu")
    updated_cell = microe_instance.export_center_cell_with_representations(
        representations=["raw_expression", "neighbor_composition", "nn_embedding"],
        model=dummy_model,
        device=dummy_device
    )
    # Check that all three keys are present.
    assert "raw_expression" in updated_cell.additional_features
    assert "neighbor_composition" in updated_cell.additional_features
    assert "nn_embedding" in updated_cell.additional_features
    # Verify the numeric values.
    np.testing.assert_array_equal(
        updated_cell.get_feature("raw_expression"),
        np.array([1.0, 2.0])
    )
    np.testing.assert_array_equal(
        updated_cell.get_feature("neighbor_composition"),
        np.array([0.5, 0.5])
    )
    np.testing.assert_array_equal(
        updated_cell.get_feature("nn_embedding"),
        np.array([0.1, 0.2, 0.3])
    )

def test_export_representations_skip_nn(microe_instance, caplog):
    """
    Test export_center_cell_with_representations when model and device are not provided.
    
    Expected:
      - The "nn_embedding" representation should be skipped (not attached) and a warning logged.
    """
    caplog.set_level(logging.WARNING)
    updated_cell = microe_instance.export_center_cell_with_representations(
        representations=["raw_expression", "neighbor_composition", "nn_embedding"],
        model=None,
        device=None
    )
    # raw_expression and neighbor_composition should be present.
    assert "raw_expression" in updated_cell.additional_features
    assert "neighbor_composition" in updated_cell.additional_features
    # nn_embedding should not be added.
    assert "nn_embedding" not in updated_cell.additional_features
    # Check that a warning was logged.
    assert any("requires model and device" in record.message for record in caplog.records)