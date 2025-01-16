#scripts/causal_inference.py
"""
Created on Thur Jan 16 12:31 2025

@author: Jiahao Zhang
@Description: 
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from spacegm.data import CellularGraphDataset
from core.causal_inference import compute_granger_causality, prepare_granger_inputs
from utils.visualization import visualize_granger_results

def load_dataset(cfg):
    dataset = CellularGraphDataset(cfg.general.data_root, **cfg.dataset)
    return dataset

def perform_causal_analysis(cfg, dataset):
    raw_dir = cfg.general.data_root + '/voronoi'
    pseudo_time_values, neighborhood_matrices, cell_type_names, biomarker_names = prepare_granger_inputs(
        dataset, raw_dir, pseudotime_file=cfg.causal_inference.pseudotime_file, cell_type_mapping=cfg.constants.CELL_TYPE_MAPPING_UPMC
    )

    p_values_df, significance_df = compute_granger_causality(
        pseudo_time=pseudo_time_values,
        neighborhood_matrices=neighborhood_matrices,
        cell_type_names=cell_type_names,
        biomarker_names=biomarker_names
    )
    return p_values_df, significance_df

def visualize_results(p_values_df, significance_df):
    visualize_granger_results(p_values_df, significance_df)

@hydra.main(config_path="../config/test", config_name="base")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Initialize and load data
    dataset = load_dataset(cfg)

    # Perform causal analysis
    p_values_df, significance_df = perform_causal_analysis(cfg, dataset)

    # Visualize results
    visualize_results(p_values_df, significance_df)

if __name__ == "__main__":
    main()
