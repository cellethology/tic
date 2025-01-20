# -*- coding: utf-8 -*-
# core/causal_inference.py
"""
Created on Wed Jan 15 18:03 2025
Last modified on [last modification date]

@author: Jiahao Zhang
@Description: This module is designed to facilitate causal inference analysis within the context of biological research,
              particularly focusing on the analysis of cellular behavior based on biomarker data. It provides a robust framework
              for applying various statistical methods to discern causal relationships from complex datasets.

              The module integrates several causal inference methods including Granger Causality, Linear Regression with Controls,
              each encapsulated within its own class. These classes inherit from a common abstract base class,
              ensuring that each method adheres to a consistent interface for analysis and visualization.

              The main functionality includes:
              - Data loading and preprocessing to prepare it for causal analysis.
              - Execution of different causal inference strategies.
              - Visualization of inference results to aid in interpretation and presentation.
"""

from abc import ABC, abstractmethod
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from adapters.space_gm_adapter import get_neighborhood_cell_ids
from statsmodels.tsa.stattools import grangercausalitytests
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from sklearn.linear_model import LinearRegression
from scipy import stats

from core.constant import ALL_CELL_TYPES, ALL_BIOMARKERS, COLUMN_MAPPING
#----------------------------------
# Helper Functions 
#----------------------------------

def load_data(file_path, column_mapping):
    # Load the data
    df = pd.read_csv(file_path)
    # Apply the column renaming mapping
    def apply_column_mapping(df, mapping):
        # Apply column renaming only for columns that exist in both the DataFrame and the mapping
        valid_mapping = {old: new for old, new in mapping.items() if old in df.columns}
        df.rename(columns=valid_mapping, inplace=True)

        # Check and log if any expected columns are missing after attempting to rename
        # missing_columns = [new for old, new in mapping.items() if old not in df.columns and new in df.columns]
        # if missing_columns:
        #     print(f"Note: Expected columns not found in the DataFrame and were not renamed: {missing_columns}")
        return df

    df = apply_column_mapping(df, column_mapping)
    return df

def load_biomarker_data(region_id, cell_id, raw_dir, biomarker: str):
    '''
    Load Cell Level Biomarker Data for a given region and cell.
    
    Args:
        region_id (str): The ID of the region to load data from.
        cell_id (str): The ID of the cell to load data for.
        raw_dir (str): Directory containing raw data files.
        biomarker (str): The specific biomarker whose data is to be loaded.
    
    Returns:
        float: The value of the biomarker for the specified cell, or 0 if not found.
    '''
    biomarker_file = os.path.join(raw_dir, f"{region_id}.expression.csv")
    biomarker_df = pd.read_csv(biomarker_file, index_col='CELL_ID') 
    if biomarker in biomarker_df.columns:
        return biomarker_df.loc[cell_id, biomarker]
    else:
        return 0 


def load_and_prepare_data(dataset, pseudotime_file, raw_dir, included_cell_types=ALL_CELL_TYPES, included_biomarkers=ALL_BIOMARKERS, sparsity_threshold=0.1, target_biomarker='PanCK'):
    """
    Load pseudotime data and compute biomarker matrices for cell neighborhoods.
    Filters out variables with high sparsity or constant values before analysis.

    Args:
        dataset: CellularGraphDataset or similar for accessing cell neighborhood data.
        pseudotime_file (str): Path to the CSV file containing pseudotime data.
        raw_dir (str): Directory containing additional cell and biomarker data.
        included_cell_types (list): Optional list of cell types to include in the analysis.
        included_biomarkers (list): Optional list of biomarkers to include in the analysis.
        sparsity_threshold (float): The threshold for filtering out sparse variables (default is 0.1, meaning >90% zeros).
        target_biomarker (str): The biomarker to be used as the dependent variable, e.g., 'PanCK'.

    Returns:
        CausalInferenceInput: Data structured for causal inference analysis.
    """
    # Load pseudotime data
    pseudotime_df = load_data(pseudotime_file, COLUMN_MAPPING)
    identifiers = []
    y_variable = []
    x_variables = {}

    # Iterate over the rows of the pseudotime data
    for _, row in pseudotime_df.iterrows():
        region_id = row['REGION_ID']
        cell_id = row['CELL_ID']
        pseudotime = row['PSEUDOTIME']


        # Load expression data for the region and cell type
        biomarker_matrix = compute_biomarker_matrix(
            dataset, region_id, cell_id, raw_dir,
            included_cell_types=included_cell_types, 
            included_biomarkers=included_biomarkers
        )

        identifiers.append([region_id, cell_id])

        # Use load_biomarker_data to get the Y variable (target biomarker value)
        y_value = load_biomarker_data(region_id, cell_id, raw_dir, target_biomarker)
        y_variable.append(y_value)

        # Set X variables (cell type and biomarker combination)
        for biomarker in included_biomarkers:
            for cell_type in included_cell_types:
                x_key = f"{cell_type}_{biomarker}"
                value = biomarker_matrix.get(biomarker, pd.Series(index=biomarker_matrix.index)).get(cell_type, 0)
                if x_key not in x_variables:
                    x_variables[x_key] = []
                x_variables[x_key].append(value)

    # Filter out variables with high sparsity or constant values
    x_variables = {k: v for k, v in x_variables.items() if (np.mean(np.array(v) != 0) > sparsity_threshold and np.unique(v).size > 1)}

    return CausalInferenceInput(identifiers=identifiers, y_variable=y_variable, x_variables=x_variables)

def compute_biomarker_matrix(dataset, region_id, center_cell_id, raw_dir, included_cell_types=None, included_biomarkers=None):
    """
    Compute the biomarker matrix for cells in a specified neighborhood.

    Args:
        dataset (CellularGraphDataset): The dataset instance.
        region_id (str): Region ID to search for.
        center_cell_id (int): Central cell ID whose neighborhood is considered.
        raw_dir (str): Directory containing cell types and biomarker data.
        included_cell_types (list): Optional list of cell types to include.
        included_biomarkers (list): Optional list of biomarkers to include.

    Returns:
        pd.DataFrame: A DataFrame representing the biomarker matrix, indexed by cell type names.
    """
    # Load cell types and expression data
    cell_types_path = f"{raw_dir}/{region_id}.cell_types.csv"
    expression_path = f"{raw_dir}/{region_id}.expression.csv"
    cell_types_df = load_data(cell_types_path, COLUMN_MAPPING)
    expression_df = load_data(expression_path, COLUMN_MAPPING)

    # Filter for neighborhood cells
    neighborhood_cell_ids = get_neighborhood_cell_ids(dataset, region_id, center_cell_id)
    neighborhood_df = cell_types_df[cell_types_df['CELL_ID'].isin(neighborhood_cell_ids)]
    expression_subset = expression_df[expression_df['CELL_ID'].isin(neighborhood_cell_ids)]

    # Merge the filtered cell type and expression data
    merged_df = pd.merge(neighborhood_df, expression_subset, on='CELL_ID', how='outer')

    # Ensure all specified cell types and biomarkers are included
    if included_cell_types is not None:
        merged_df['CELL_TYPE'] = merged_df['CELL_TYPE'].fillna(pd.Series(included_cell_types))
    if included_biomarkers is not None:
        for biomarker in included_biomarkers:
            if biomarker not in merged_df:
                merged_df[biomarker] = 0

    # Compute average biomarker expressions for each cell type
    biomarker_columns = [col for col in merged_df.columns if col not in ['CELL_ID', 'REGION_ID', 'CELL_TYPE']]
    biomarker_matrix = merged_df.groupby('CELL_TYPE')[biomarker_columns].mean().fillna(0)

    return biomarker_matrix

#----------------------------------
# Data Class for Causal Inference
#----------------------------------

class CausalInferenceInput(BaseModel):
    identifiers: List[List[Any]] = Field(..., description="Unique identifiers for each study object, e.g., region and cell IDs.")
    y_variable: List[Any] = Field(..., description="The dependent variable for causal inference, e.g., PSEUDOTIME.")
    x_variables: Dict[str, List[Any]] = Field(..., description="Independent variables for causal inference, structured by cell type and biomarker.")

#----------------------------------
# Core: Casual Inference
#----------------------------------

class CausalInferenceMethod(ABC):
    @abstractmethod
    def analyze(self, data: CausalInferenceInput) -> Any:
        """
        Perform causal analysis on the provided data.
        Args:
            data (CausalInferenceInput): Data structured for causal inference analysis.
        Returns:
            Any: Results of the causal analysis, structure depending on the method used.
        """
        pass

    @abstractmethod
    def visualize(self, results: pd.DataFrame):
        """
        Plot the results of the causal analysis.
        Args:
            results (pd.DataFrame): DataFrame containing the results of a causal analysis.
        """
        pass

class GrangerCausality(CausalInferenceMethod):
    def analyze(self, data: CausalInferenceInput) -> pd.DataFrame:
        results = {}
        for x_key, values in data.x_variables.items():
            combined_data = pd.DataFrame({'Y': data.y_variable, 'X': values})
            try:
                test_result = grangercausalitytests(combined_data, maxlag=2, verbose=False)
                min_p_value = min(test[1][0]['ssr_ftest'][1] for test in test_result.items())
                results[x_key] = min_p_value
            except Exception as e:
                print(f"Error processing {x_key}: {e}")
                continue

        return pd.DataFrame.from_dict(results, orient='index', columns=['P-Value'])

    def visualize(self, results: pd.DataFrame):
        if results.empty:
            print("No results to display.")
            return

        results_sorted = results.sort_values(by='P-Value').head(10)  # Visualize the top 10 results by P-Value
        results_sorted.plot(kind='bar', legend=False)
        plt.title('Top 10 Granger Causality Results')
        plt.xlabel('Variable Pairs')
        plt.ylabel('P-Value')
        plt.show()

class LinearRegressionWithControls(CausalInferenceMethod):
    def analyze(self, data: CausalInferenceInput) -> pd.DataFrame:
        results = {}
        model = LinearRegression()
        for x_key, values in data.x_variables.items():
            if pd.Series(values).nunique() <= 1:  # Avoid variables with constant values
                continue

            X = pd.DataFrame({k: v for k, v in data.x_variables.items() if k != x_key})  # Control variables
            y = pd.Series(values)
            if X.empty:
                continue  # Avoid fitting a model with no explanatory variables

            model.fit(X, y)
            p_values = stats.t.ppf(1 - 0.05, df=len(y) - X.shape[1] - 1)  # p-values calculation

            results[x_key] = {'Coefficient': model.coef_, 'P-Value': p_values}

        return pd.DataFrame(results).T  

    def visualize(self, results: pd.DataFrame):
        if results.empty:
            print("No results to display.")
            return

        # Sorting the P-Values and taking top 10 for visualization
        try:
            sorted_results = results.sort_values(by='P-Value', ascending=True).head(10)
            sorted_results['P-Value'].plot(kind='bar')
            plt.title('Top 10 Linear Regression Results by P-Value')
            plt.xlabel('Variable Pairs')
            plt.ylabel('P-Value')
            plt.show()
        except Exception as e:
            print(f"Error visualizing results: {e}")



def run_causal_inference_analysis(data: CausalInferenceInput, method_type='GrangerCausality'):
    if method_type == 'GrangerCausality':
        method = GrangerCausality()
    elif method_type == 'LinearRegressionWithControls':
        method = LinearRegressionWithControls()
    else:
        raise ValueError("Unsupported method type provided.")

    # Perform analysis
    results = method.analyze(data)
    method.visualize(results)

    return results

