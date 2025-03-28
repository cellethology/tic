"""
Tests for the Cell module.

Functionalities to test:
- Biomarkers: attribute access, error handling for missing biomarkers, and string representation.
- Cell: proper initialization, getting biomarker values (including non-existent ones), and managing additional features.
"""

import pytest

from utils.mock_data import create_mock_biomarkers, create_mock_cell


def test_biomarkers_getattr_success():
    b = create_mock_biomarkers(PanCK=1.5, CD3=2.0)
    assert b.PanCK == 1.5
    assert b.CD3 == 2.0

def test_biomarkers_getattr_failure():
    # Create a Biomarkers object without defaults so "CD3" is missing.
    b = create_mock_biomarkers(include_defaults=False, PanCK=1.5)
    with pytest.raises(AttributeError):
        _ = b.CD3

def test_cell_get_biomarker():
    # Create a cell with Biomarkers that do not include "CD3"
    cell = create_mock_cell(cell_id="TestCell", biomarkers=create_mock_biomarkers(include_defaults=False, PanCK=1.5))
    assert cell.get_biomarker("PanCK") == 1.5
    # When biomarker "CD3" does not exist, get_biomarker should print a warning and return None.
    assert cell.get_biomarker("CD3") is None

def test_cell_feature_add_and_get():
    cell = create_mock_cell(cell_id="TestCell")
    cell.add_feature("geneA", 3.14)
    assert cell.get_feature("geneA") == 3.14