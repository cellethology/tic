#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_microe_causal.py

Demonstration script for:
1. Loading an external list of Cell objects, each with pseudo-time, region_id, and cell_id.
2. Sorting them by pseudo-time in ascending order.
3. Mapping each cell to a MicroE object in a MicroEDataset (based on matching region_id and cell_id).
4. Creating a time-ordered list of MicroE objects.
5. Building a DataFrame for causal inference from those MicroE objects.
6. Running a causal analysis method on the sorted data.

"""
import os
import logging
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
import sys

from core.data.dataset import MicroEDataset
from core.data.cell import Cell
from core.constant import ALL_BIOMARKERS, ALL_CELL_TYPES
from core.causal.base import BaseCausalMethod
from core.causal.causal_input import CausalInput
from core.causal.repo.granger_causality import GrangerCausalityMethod
from core.data.microe import MicroE
from utils.extract_representation import get_region_ids_from_raw
from utils.plot import plot_top_x_effects, plot_x_effect_heatmap

logger = logging.getLogger(__name__)


def load_and_sort_cells(cell_pt_file: str) -> List[Cell]:
    """Load cells with pseudotime from a given file."""
    if not os.path.exists(cell_pt_file):
        raise FileNotFoundError(f"File not found: {cell_pt_file}")

    cells = torch.load(cell_pt_file)
    for cell in cells:
        if cell.get_feature("pseudotime") is None:
            raise ValueError(f"Missing 'pseudotime' for cell {cell.cell_id}")
    logger.info(f"Loaded {len(cells)} cells from {cell_pt_file}")
    sorted(cells, key=lambda cell: cell.get_feature("pseudotime") or float('inf'))
    logger.info("Sorted cells by pseudo-time.")
    return cells


def gather_microe_in_pseudotime(
    dataset: MicroEDataset,
    sorted_cells: List[Cell]
) -> List[MicroE]:
    """
    For each cell (already sorted by pseudotime), find the corresponding MicroE
    in the MicroEDataset (matching region_id and cell_id). Collect these in ascending
    pseudo-time order.

    Args:
        dataset (MicroEDataset): The dataset containing pre-built MicroEs.
        sorted_cells (List[Cell]): Cells sorted in ascending pseudo-time.

    Returns:
        List[MicroE]: MicroEs matched and sorted by the cell's pseudo-time.
                      If no MicroE is found for a cell, that cell is skipped.
    """
    microe_list = []
    for c in sorted_cells:
        region_id = c.tissue_id
        center_id = c.cell_id
        try:
            me = dataset.get_microE(region_id, center_id)
            microe_list.append(me)
        except FileNotFoundError:
            logger.warning(f"MicroE not found for region={region_id}, cell_id={center_id}. Skipping.")
        except Exception as e:
            logger.error(f"Error retrieving MicroE for region={region_id}, cell_id={center_id}: {e}")
            continue

    logger.info(f"Gathered {len(microe_list)} MicroEs in ascending pseudo-time order.")
    return microe_list


def microe_list_to_dataframe(
    microe_list: List[MicroE],
    y_biomarkers: List[str],
    x_biomarkers: Optional[List[str]] = None,
    x_cell_types: Optional[List[str]] = None,
    drop_nan_cols: bool = True
) -> pd.DataFrame:
    """
    Convert a list of MicroE objects into a DataFrame with columns for X (neighbor biomarkers)
    and Y (center cell biomarkers). The order of rows will correspond to the order in `microe_list`,
    which should already be sorted by pseudo-time.

    Args:
        microe_list (List[MicroE]): MicroE objects in ascending pseudo-time order.
        y_biomarkers (List[str]): Which biomarkers in the center cell to treat as Y.
        x_biomarkers (Optional[List[str]]): Biomarkers for X. Defaults to ALL_BIOMARKERS if None.
        x_cell_types (Optional[List[str]]): Cell types for X. Defaults to ALL_CELL_TYPES if None.
        drop_nan_cols (bool): If True, drop columns that are fully NaN, then fill partial NaNs with 0.

    Returns:
        pd.DataFrame: DataFrame with each row as one MicroE, preserving input order.
                      Columns: [X_1, X_2, ..., Y_1, Y_2, ...].
    """
    if x_biomarkers is None:
        x_biomarkers = ALL_BIOMARKERS
    if x_cell_types is None:
        x_cell_types = ALL_CELL_TYPES

    data_rows = []
    X_labels, Y_labels = None, None

    for me in microe_list:
        X, Y, cur_X_labels, cur_Y_labels = me.prepare_for_causal_inference(
            y_biomarkers=y_biomarkers,
            x_biomarkers=x_biomarkers,
            x_cell_types=x_cell_types
        )
        row = np.concatenate([X, Y])
        data_rows.append(row)

        # Capture labels from the first MicroE
        if X_labels is None:
            X_labels = cur_X_labels
            Y_labels = cur_Y_labels

    if not data_rows:
        logger.warning("No MicroE data was collected. Returning empty DataFrame.")
        return pd.DataFrame()

    column_names = list(X_labels) + list(Y_labels)
    df = pd.DataFrame(data_rows, columns=column_names)

    if drop_nan_cols:
        df = df.dropna(axis=1, how="all")
        df = df.fillna(0.0)
    else:
        df = df.fillna(0.0)

    return df


def run_causal_analysis_multiple_y(
    df: pd.DataFrame,
    y_biomarkers: List[str],
    causal_method: BaseCausalMethod
) -> pd.DataFrame:
    """
    For each Y biomarker, iterate over X columns as a potential "treatment."
    Fits and estimates the causal effect, returning a summary DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns for X, Y in ascending pseudo-time order.
        y_biomarkers (List[str]): Column names corresponding to outcome variables.
        causal_method (BaseCausalMethod): A method implementing fit() and estimate_effect().

    Returns:
        pd.DataFrame: Each row is (y_biomarker, x_variable, estimated_effect, p_value, ...).
    """
    x_cols = [col for col in df.columns if col not in y_biomarkers]

    results = []
    for y_col in y_biomarkers:
        for x_col in x_cols:
            cinput = CausalInput(
                data=df,
                treatment_col=x_col,
                outcome_col=y_col,
                covariates=[xc for xc in x_cols if xc != x_col]
            )
            causal_method.fit(cinput)
            effect_est = causal_method.estimate_effect(cinput)

            results.append({
                "y_biomarker": y_col,
                "x_variable": x_col,
                "estimated_effect": effect_est.get("estimated_effect"),
                "p_value": effect_est.get("p_value"),
                "additional_info": effect_est.get("additional_info", None)
            })

    return pd.DataFrame(results)


def main():
    """
    Example usage:
    1. Load an external .pt file containing Cells with pseudo-time, region_id, cell_id.
    2. Sort them by pseudo-time.
    3. Build / load a MicroEDataset containing pre-built MicroEs.
    4. Match each sorted cell to its MicroE in the dataset, gather in ascending pseudo-time.
    5. Convert them to a DataFrame for causal inference.
    6. Run your chosen causal method (e.g. Granger Causality).
    """

    # 1) Load & sort cells by pseudotime
    cell_pt_file = "/Users/zhangjiahao/Project/tic/results/experiment/experiment_20250311_111213/cells_with_pseudotime.pt"
    sorted_cells = load_and_sort_cells(cell_pt_file)

    # 2) Instantiate MicroEDataset
    root_dir = "/Users/zhangjiahao/Project/tic/data/example"
    region_ids = get_region_ids_from_raw(root_dir)
    dataset = MicroEDataset(root=root_dir, region_ids=region_ids, k=3)
    # This dataset has pre-built MicroEs, each accessible by region+cell_id

    # 3) Gather MicroEs in ascending pseudo-time
    sorted_microes = gather_microe_in_pseudotime(dataset, sorted_cells)

    # 4) Convert MicroE list to a DataFrame (X, Y)
    y_biomarkers = ["PanCK", "aSMA"]  # example
    df = microe_list_to_dataframe(sorted_microes, y_biomarkers=y_biomarkers, drop_nan_cols=True)

    if df.empty:
        logger.error("No valid MicroE data matched the pseudo-time cells. Exiting.")
        sys.exit(1)

    # 5) Choose a causal method (example: GrangerCausalityMethod)
    gc_method = GrangerCausalityMethod(maxlag=2)

    # 6) Run analysis across each (Y, X) pair
    results_df = run_causal_analysis_multiple_y(df, y_biomarkers, gc_method)
    print("Causal inference results:")
    print(results_df.head())
    plot_top_x_effects(results_df,y_biomarkers=y_biomarkers,key='p_value',ascending=True)
    plot_x_effect_heatmap(results_df,y_biomarkers = "PanCK", key='p_value')
    print("Plotted top X effects and heatmap.")
    print("Done.")

if __name__ == "__main__":
    main()