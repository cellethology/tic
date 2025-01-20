# scripts/causal_inference.py
"""
Created on Thur Jan 16 12:31 2025

@author: Jiahao Zhang
@Description: Run causal inference analysis based on inferred pseudotime.
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from spacegm.data import CellularGraphDataset
from core.causal_inference import load_and_prepare_data, run_causal_inference_analysis

def load_dataset(cfg):
    """
    Initialize and load the dataset.

    Args:
        cfg (DictConfig): Configuration object with dataset parameters.

    Returns:
        CellularGraphDataset: Loaded dataset instance.
    """
    dataset = CellularGraphDataset(cfg.general.data_root, **cfg.dataset)
    return dataset

def perform_causal_analysis(cfg, dataset):
    """
    Perform causal inference analysis based on pseudotime data.

    Args:
        cfg (DictConfig): Configuration object.
        dataset (CellularGraphDataset): Loaded dataset instance.
    """
    # Load and prepare data for causal analysis
    causal_input = load_and_prepare_data(
        dataset,
        pseudotime_file=cfg.causal.pseudotime_file,
        raw_dir=cfg.general.raw_dir,
        included_cell_types=cfg.causal.included_cell_types,
        included_biomarkers=cfg.causal.included_biomarkers,
        sparsity_threshold=cfg.causal.sparsity_threshold,
        target_biomarker=cfg.causal.target_biomarker,
    )

    # Run causal inference analysis
    results = run_causal_inference_analysis(causal_input, method_type=cfg.causal.method)

    # Save results to CSV
    output_dir = cfg.causal.visualization.output_dir
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"causal_results_{cfg.causal.method}.csv")
    results.to_csv(results_path, index=True)

    print(f"Causal inference results saved to {results_path}")

@hydra.main(config_path="../config/pseudotime", config_name="new")
def main(cfg: DictConfig):
    """
    Main function to run the causal inference pipeline.

    Args:
        cfg (DictConfig): Configuration object.
    """
    print(OmegaConf.to_yaml(cfg))

    # Step 1: Initialize and load data
    dataset = load_dataset(cfg)

    # Step 2: Perform causal analysis
    perform_causal_analysis(cfg, dataset)

if __name__ == "__main__":
    main()
